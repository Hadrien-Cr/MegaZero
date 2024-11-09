# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
"""
This script implements the Evolutionary MCTS (EMCTS).
This is a variant of MCTS where the transitions are not actions but are instead mutations of a long sequence of actions.


Class Enode: Represents a Node of the tree search
    Attributes:
        seq: the sequence of action that it represents
        m (eg Mutation(tau=1, a=2)): the mutation that it represents (seq is obtained by switching seq[1] to a=2 in the parent sequence) 
        _children: list of the children of the node
        n,v,q,p,e: statistics on the visits, the value, the qvalue, the prior, and the "terminality" of the node
        player: represents to which player is associated the mutation
    Methods:
        add_children(self, valid_mutations, mutate_prior, player_history,num_players): uses the valid mutations and the mutate prior to create children node
        best_child(self, fpu_reduction, cpuct): uses UCT formula to select the best node.


Class EMCTS: Object that performs the search
    Attributes:
        - seq_length: the length of each sequence (padded if needed)
        - _root, _curnode, _path: root, current node, and current path constructed
        - policy_history: array that keeps track of the number of visits of the mutations
        - n_phases: Maximum number of mutations update to apply to the root
        - phase: Current number of mutations applied to the root
        - turn_completed: if True, then all phases are used and the search should be reset
    Methods:
        - find_leaf(self, gs): Method for selection + expansion
            If the root sequence is not initialized yet, play it to its end and return the leaf state
            If the root sequence is initialized, look for the best mutations of the sequences until the current node has not been visited, 
            expand it, play the sequence and return the leaf state.
       -  process_results(self, gs, value, pi, add_root_noise, add_root_temp):  for backpropagating the vectors (value, pi)
            If the root sequence is not initialized yet, use pi to extend it.
            Else, perform the standard backpropagation from MCTS algorithm is
        - update_turn(self, gs): Method to update the root by taking the best mutation found, increments the phase
        - get_results(self, gs): Method to extract the results of the search
            Returns the root turn sequence, the state visited by playing this sequence, and the policy history normalized
"""
from libc.math cimport sqrt

import numpy as np
cimport numpy as np
from alphazero.utils import dotdict
import copy
import time
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
_DRAW_VALUE = 0.5
PAD = -1
np.seterr(all='raise')


cdef class Mutation:
    cdef public int tau
    cdef public int a
    def __init__(self, int tau, int a):
        self.tau = tau
        self.a = a
    def __eq__(self, Mutation other):
        return(self.tau == other.tau) and (self.a == other.a)
    def __repr__(self):
        return 'Mutation({}, {})' \
            .format(self.tau, self.a)

# @cython.auto_pickle(True)
cdef class ENode:
    cdef public list _children
    cdef public list seq
    cdef public Mutation m
    cdef public np.ndarray e
    cdef public float q
    cdef public float v
    cdef public int n
    cdef public float p
    cdef public int player

    def __init__(self, list seq, Mutation m, int num_players, float p, int player):
        self._children = []
        self.seq = seq
        self.m = m
        self.e = np.zeros(num_players, dtype=np.uint8)
        self.q = 0
        self.v = 0
        self.n = 0
        self.p = p
        self.player = player

    def __repr__(self):
        return 'ENode(m={}, seq={}, e={}, q={}, v={}, p={}, n={})' \
            .format(self.m, self.seq, self.e, self.q, self.v, self.p, self.n)

    cdef void add_children(self, np.ndarray valid_mutations, 
                                np.ndarray mutate_prior,
                                list player_history,
                                int num_players):
        """
        Adds mutated childrens by providing an array valid_mutations[t,a] of the mutation that can be performed and the mutate_prior[t,a] array
        """
        cdef int tau, a, player
        assert len(self._children) == 0, f"{self}{self._children}"

        for tau in range(len(self.seq)):
            player = player_history[tau]
            self._children.extend([ENode(seq = [],
                                        m = Mutation(tau,a), 
                                        num_players = num_players,
                                        p = mutate_prior[tau, a],
                                        player = player) 
                                    for a, valid in enumerate(valid_mutations[tau]) if valid])
        # shuffle children
        #np.random.shuffle(self._children)
        assert len(self._children)>0 or self.e.any(),  f"No children has been created from non terminal node {self} with {np.sum(valid_mutations)} valid_mutations."

    cdef float uct(self, float sqrt_parent_n, float fpu_value, float cpuct):
        return (fpu_value if self.n == 0 else self.q) + cpuct * self.p * sqrt_parent_n / (1 + self.n)

    cdef ENode best_child(self, float fpu_reduction, float cpuct):
        cdef ENode c
        cdef float seen_policy = sum([c.p for c in self._children if c.n > 0])
        cdef float fpu_value = self.v - fpu_reduction * sqrt(seen_policy)
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self.n)
        cdef float uct
        child = None

        for c in self._children:
            uct = c.uct(sqrt_n, fpu_value, cpuct)
            if uct > cur_best:
                cur_best = uct
                child = c
            elif not self.e[self.player] and c.e[self.player]:
                return c
        return child



# @cython.auto_pickle(True)
cdef class EMCTS:
    cdef public float root_noise_frac
    cdef public float root_temp
    cdef public float min_discount
    cdef public float fpu_reduction
    cdef public float cpuct
    cdef public int _num_players
    
    cdef public ENode _root
    cdef public ENode _curnode
    cdef public list _path

    cdef public np.ndarray policy_history
    cdef public list player_history
    cdef public np.ndarray mutate_prior
    cdef public int seq_length
    cdef public int phase
    cdef public bint turn_completed
    cdef public int n_phases
    cdef public int depth
    cdef public int max_depth
    cdef public int _discount_max_depth
    
    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct
        self.n_phases = args.emcts_bb_phases

        self._num_players = args._num_players
        self._root = ENode([], Mutation(-1,-1), self._num_players, p = 1, player = 0)
        self._curnode = self._root
        self._path = []

        self.phase = 0
        self.turn_completed = False

        self.mutate_prior = None # An array that gives the prior probability mutate_prior[t,a] to mutate the t-th to action a 
        self.player_history = []
        self.seq_length = args.emcts_horizon

        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    def __repr__(self):
        return 'EMCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, _num_players={}, ' \
               '_root={}, _curnode={}, _path={}, depth={}, max_depth={})' \
            .format(self.root_noise_frac, self.root_temp, self.min_discount,
                    self.fpu_reduction,self.cpuct, self._num_players, self._root,
                    self._curnode, self._path, self.depth, self.max_depth)

    cpdef void reset(self):
        self._root = ENode([], Mutation(-1,-1), self._num_players, p = 1, player = 0)
        self._curnode = self._root
        self._path = []

        self.phase = 0
        self.turn_completed = False

        self.player_history = []
        self.mutate_prior = None # An array that gives the prior probability mutate_prior[t,a] to mutate the t-th to action a 
        
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    # def __reduce__(self):
    #   return rebuild_mcts, (self._root._players, self.cpuct, self._root, self._curnode, self._path)

    cpdef void search(self, object gs, object nn, int sims, bint add_root_noise, bint add_root_temp):
        assert (self.seq_length%gs.d) == 0
        cdef float[:] v
        cdef float[:] p
        self.max_depth = 0
        assert not gs.win_state().any()
        assert gs.micro_step == 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            p, v = nn(leaf.observation())
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void raw_search(self, object gs, int sims, bint add_root_noise, bint add_root_temp):
        assert (self.seq_length%gs.d) == 0
        cdef Py_ssize_t policy_size = gs.action_size()
        cdef float[:] v = np.zeros(gs.num_players() + 1, dtype=np.float32)  #np.full((value_size,), 1 / value_size, dtype=np.float32)
        cdef float[:] p = np.full(policy_size, 1, dtype=np.float32)
        self.max_depth = 0
        assert not gs.win_state().any()
        assert gs.micro_step == 0
        
        for _ in range(sims):
            leaf = self.find_leaf(gs)
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)
    
    def get_results(self, object gs):

        assert self.turn_completed
        assert np.sum(self.policy_history)>0

        action_history, pi, state_history = [], [], []
        
        for t, a in enumerate(self._root.seq):
            assert a!= PAD, f"{t,a} {self._root.seq}"
            action_history.append(a)
            state_history.append(gs.clone())
            pi.append(self.policy_history[t]/np.sum(self.policy_history[t]))
            gs.play_action(a)
            if gs.micro_step == 0 or gs.win_state().any():
                break 
        assert gs.micro_step == 0 or gs.win_state().any()
        return action_history, pi, state_history

    cpdef object find_leaf(self, object gs):
        """
        Find the next node to evaluate and return the game state

        Pseudo code:
        if the root sequence is not completed yet:
            Play Root Sequence to update gs
            Return gs

        else:
            Find the best mutations M (using self._curnode._best_child) until the resulting node has not yet been visited
            node.sequence = mutate(previous node, m)
            Play node.sequence to update gs
            Return gs
        """
        self.depth = 0
        self._curnode = self._root
        cdef list parent_seq = self._root.seq
        cdef object leaf = gs.clone()
        assert not gs.win_state().any(), "Impossible to call find_leaf on terminal node"
        assert gs.micro_step == 0,  f"{self} {leaf} "
        # If the root sequence is not initialized
        if len(self._root.seq) < self.seq_length and not self._root.e.any():
            for a in self._root.seq:
                assert not leaf.win_state().any(), f"{self} {leaf} {a}"
                if a!= PAD: leaf.play_action(a)

        # Else if the root sequence is initialized, look for mutations
        else:
            while self._curnode.n > 0 and not self._curnode.e.any():
                self._path.append(self._curnode)
                parent_seq = self._curnode.seq
                self._curnode = self._curnode.best_child(self.fpu_reduction, self.cpuct)
                self.depth += 1

            if self.depth > self.max_depth:
                self.max_depth = self.depth
                self._discount_max_depth = self.depth
            
            if self._curnode.n == 0:
                if len(self._path)==0: self._curnode.seq = []
                valids = self.mutate_and_play(state = leaf, 
                                            parent_seq = parent_seq,
                                            node = self._curnode)
                self._curnode.e = leaf.win_state()
                self._curnode.add_children( valid_mutations = valids, 
                                            mutate_prior = self.mutate_prior,
                                            num_players = self._num_players,
                                            player_history = self.player_history)

            if self.depth == 1 and not self._curnode.e.any() and self._root.e.any():
                self.update_turn(gs, self._curnode.m)

        return(leaf)

    cpdef np.ndarray mutate_and_play(self, object state, list parent_seq, ENode node):
        '''
        Returns valid_mutations
        '''
        assert node.n == 0
        assert len(node.seq) == 0
        cdef np.ndarray valids = np.zeros((len(parent_seq), state.action_size()), dtype=np.float32)
        player = state._player
        t = 0
        while t < len(parent_seq):
            if state.win_state().any(): 
                valid_moves = state.valid_moves()
                valids[t] = valid_moves
                break
            else:
                a = node.m.a if (t == node.m.tau and node.m.a != PAD) else parent_seq[t]
                valid_moves = state.valid_moves()
                
                if not valid_moves[a]:
                    a = np.argmax(self.mutate_prior[t]*state.valid_moves()).item()
                    self.policy_history[t, a] +=1
                valids[t] = valid_moves
                valids[t, a] = 0 
                state.play_action(a)  
                node.seq.append(a)
                
            t+=1

            # pad when the end of turn is reached
            if state.micro_step == 0 and t < len(parent_seq):
                while (t%state.d) != 0:
                    node.seq.append(PAD)
                    t+=1
                player = state._player
        return valids        
    
    cpdef object reconstruct_leaf(self, gs):
        
        cdef object leaf = gs.clone()
        
        if len(self._root.seq) < self.seq_length and not self._root.e.any():
            for a in self._root.seq:
                assert not leaf.win_state().any(), f"{self} {leaf} {a}"
                if a!= PAD: 
                    leaf.play_action(a)
        
        return leaf

    cpdef void process_results_from_init(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        """
        Should be called at very beginning of the search, when the root sequence is not completed
        pi should help store the self.mutate_prior, which will be shared for all nodes.
        """

        if self.mutate_prior is None:
            self.mutate_prior = np.zeros((self.seq_length, gs.action_size()), dtype=np.float32)
        
        if self.policy_history is None:
            self.policy_history = np.zeros((self.seq_length, gs.action_size()), dtype=np.int64)
        
        assert self._curnode == self._root

        tau = len(self._root.seq)
        a = self.osla_test(gs)
        self.mutate_prior[tau] = pi
        
        if a == PAD: # no winning move found
            pi *= np.asarray(gs.valid_moves(), dtype=np.float32)
            # add root temperature
            if add_root_temp:
                pi = np.asarray(pi) ** (1.0 / self.root_temp)
                # re-normalize
                pi /= np.sum(pi)
                    
            a = np.random.choice(len(pi), p = pi/np.sum(pi))
        
        if tau < gs.d: 
            self.policy_history[tau,a] += 1
        
        self._root.seq.append(a)
        self.player_history.append(gs._player)
        gs.play_action(a)

        if gs.micro_step == 0:
            # Pad the sequences at the of a turn so that a turn is always of length d
            tau+=1
            n = len(self._root.seq)
            while tau%gs.d != 0 and tau<self.seq_length:
                self.player_history.append(gs._player)
                self._root.seq.append(PAD)
                self.mutate_prior[tau] = self.mutate_prior[n-1]

        self._root.e = gs.win_state()


    cpdef void process_results_from_mutations(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        """
        Should be called when backpropagating along mutations. value, pi refers to the NN-evaluation of the state after playing the partial root_sequence
        """

        cdef Py_ssize_t num_players = gs.num_players()
        cdef ENode parent
        cdef float v
        cdef float discount
        cdef int i = 0

        if self._curnode.e.any():
            value = np.array(self._curnode.e, dtype=np.float32)

        while self._path:
            parent = self._path.pop()
            v = self._get_value(value, parent.player, num_players)

            # apply discount only to current node's Q value
            discount = (self.min_discount ** (i / self._discount_max_depth))
            if v < _DRAW_VALUE:
                # (1 - discount) + 1 to invert it because bad values far away
                # are better than bad values close to root
                discount = 2 - discount
            elif v == _DRAW_VALUE:
                # don't discount value in the rare case that it is a precise draw (0.5)
                discount = 1

            # scale value to the range [-1, 1]
            # v = 2 * v * discount - 1

            self._curnode.q = (self._curnode.q * self._curnode.n + v * discount) / (self._curnode.n + 1)
            if self._curnode.n == 0:
                self._curnode.v = self._get_value(value, self._curnode.player, num_players)  # * 2 - 1
            self._curnode.n += 1

            m = self._curnode.m
            if m.tau <gs.d: 
                self.policy_history[m.tau, m.a] += 1

            self._curnode = parent
            i += 1

        self._root.n += 1

    cpdef void process_results(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        if len(self._root.seq) < self.seq_length and not self._root.e.any():
            self.process_results_from_init(gs, value, pi, add_root_noise, add_root_temp)
        else:
            self.process_results_from_mutations(gs, value, pi, add_root_noise, add_root_temp)

    cpdef float _get_value(self, float[:] value, Py_ssize_t player, Py_ssize_t num_players):
        if value.size > num_players:
            return value[player] + value[num_players] / num_players
        else:
            return value[player]


    cpdef float value(self, bint average=False):
        """
        Get the value of the current root node in the range [0, 1]
        by looking at the max value of child nodes (or averaging them).
        """
        cdef float value = 0
        cdef ENode c
        
        if average:
            value = sum([c.q for c in self._root._children if c.n > 0]) / len(self._root._children)
        
        else:
            for c in self._root._children:
                if c.q > value and c.n > 0:
                    value = c.q
            
        return value

    cpdef int osla_test(self, object gs):
        valid_moves = gs.valid_moves()
        cdef list win_move_set = []
        
        for move, valid in enumerate(valid_moves):
            if not valid: continue

            new_state = gs.clone()
            new_state.play_action(move)
            ws = new_state.win_state()
            if ws[gs.player]:
                win_move_set.append(move)

        if len(win_move_set) > 0:
            a = np.random.choice(win_move_set).item()
            return a
        else:
            return PAD

    cpdef void update_root(self, object gs, Mutation m):
        """
        Should be called at the end of every bridge burning phase
        Changes _root to the new root
        """
        cdef ENode c
        assert len(self._root._children)>0, f"Root {self._root} has no children"
        
        cdef float seen_policy = sum([c.p for c in self._root._children if c.n > 0])
        cdef float fpu_value = self._root.v - 0 * sqrt(seen_policy)
        cdef float cur_best = -float('inf')
        cdef float sqrt_n = sqrt(self._root.n) 
        for c in self._root._children:
            if c.m == m:
                self._root = c
                return
        raise ValueError, "Invalid mutation selected"

    cpdef void update_turn(self, object gs, float temp):
        assert self.phase <self.n_phases, "All phases are done"
        self.phase += 1

        if self.phase >= self.n_phases:
            self.turn_completed = True
        if len(self._root._children)>0 and np.sum(self.counts(gs))>0:
            # draw the best mutation found
            policy_of_mut = self.probs(gs, temp)
            idx_of_mut = np.random.choice(len(policy_of_mut), p=policy_of_mut)        
            m = Mutation(tau = idx_of_mut//gs.action_size(), a = idx_of_mut%gs.action_size())
            self.update_root(gs, m)
        else:
            pass

    cpdef np.ndarray probs(self, object gs, float temp=1.0):
        cdef float[:] counts = np.array(self.counts(gs), dtype=np.float32)
        cdef np.ndarray[dtype=np.float32_t, ndim=1] probs
        cdef Py_ssize_t best_action

        assert np.sum(self.counts(gs))>0
        
        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

        try:
            probs = (counts / np.sum(counts)) ** (1.0 / temp)
            probs /= np.sum(probs)
            return probs

        except OverflowError:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs

    cpdef int[:] counts(self, object gs):
        cdef int[:] counts = np.zeros(self.seq_length * gs.action_size(), dtype=np.int32)
        cdef ENode c
        cdef int idx_of_mut
        assert len(self._root._children) > 0
        
        for c in self._root._children:
            idx_of_mut = c.m.a + c.m.tau*gs.action_size()
            counts[idx_of_mut] = c.n
        return np.asarray(counts)

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        cdef float[:] noise = np.array(np.random.dirichlet(
            [NOISE_ALPHA_RATIO / (num_valid_moves+1)] * num_valid_moves
        ), dtype=np.float32)
        cdef ENode c
        cdef float n

        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1 - self.root_noise_frac) + self.root_noise_frac * n
    
