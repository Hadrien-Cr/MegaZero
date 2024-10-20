# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

from libc.math cimport sqrt

import numpy as np
cimport numpy as np
from alphazero.utils import dotdict
import copy

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

NOISE_ALPHA_RATIO = 10.83
_DRAW_VALUE = 0.5

np.seterr(all='raise')


cdef class Mutation:
    cdef public int tau
    cdef public int a
    def __init__(self, int tau, int a):
        self.tau = tau
        self.a = a
    def __eq__(self, Mutation other):
        self.tau = other.tau
        self.a = other.a
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

    cdef void add_children(self, valid_mutations, 
                                mutate_prior,
                                list player_history,
                                int num_players):
        """
        Adds mutated childrens by providing an array moves_to_mutate[t,a] of the mutation that can be performed and the mutate_prior[t,a] array
        """
        for tau in range(len(self.seq)):
            self._children.extend([ENode(seq = self.seq,
                                        m = Mutation(tau,a), 
                                        num_players = num_players,
                                        p = mutate_prior[tau,a],
                                        player = player_history[tau]) 
                                        for a, valid in enumerate(valid_mutations[tau]) if valid ])
        # shuffle children
        np.random.shuffle(self._children)

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

        return child
        

"""
def rebuild_mcts(num_players, cpuct, root, curnode, path):
    mcts = MCTS(num_players, cpuct)
    mcts.cpuct = cpuct
    mcts._root = root
    mcts._curnode = curnode
    mcts.path = path
    return mcts
"""


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

    cdef public list state_history
    cdef public np.ndarray policy_history
    cdef public list action_history
    cdef public list player_history
    cdef public np.ndarray mutate_prior
    cdef public int seq_length

    cdef public int depth
    cdef public int max_depth
    cdef public int _discount_max_depth
    
    def __init__(self, args: dotdict):
        self.root_noise_frac = args.root_noise_frac
        self.root_temp = args.root_policy_temp
        self.min_discount = args.min_discount
        self.fpu_reduction = args.fpu_reduction
        self.cpuct = args.cpuct

        self._num_players = args._num_players
        self._root = ENode([], Mutation(-1,-1), self._num_players, p = 1, player = 0)
        self._curnode = self._root
        self._path = []

        self.state_history = []
        self.policy_history = None # An array that stores the count of the mutations that have been made
        self.action_history = []  # Stores the best sequence found
        self.player_history = []
        self.mutate_prior = None # An array that gives the prior probability mutate_prior[t,a] to mutate the t-th to action a 
        self.seq_length = args.emcts_horizon

        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    def __repr__(self):
        return 'MCTS(root_noise_frac={}, root_temp={}, min_discount={}, fpu_reduction={}, cpuct={}, _num_players={}, ' \
               '_root={}, _curnode={}, _path={}, depth={}, max_depth={})' \
            .format(self.root_noise_frac, self.root_temp, self.min_discount,
                    self.fpu_reduction,self.cpuct, self._num_players, self._root,
                    self._curnode, self._path, self.depth, self.max_depth)

    cpdef void reset(self):
        self._root = ENode([], Mutation(-1,-1), self._num_players, p = 1, player = 0)
        self._curnode = self._root
        self._path = []

        self.state_history = []
        self.policy_history = None  # An array that stores the count of the mutations that have been made
        self.action_history = []
        self.player_history = []
        self.mutate_prior = None # An array that gives the prior probability mutate_prior[t,a] to mutate the t-th to action a 
        
        self.depth = 0
        self.max_depth = 0
        self._discount_max_depth = 0

    # def __reduce__(self):
    #   return rebuild_mcts, (self._root._players, self.cpuct, self._root, self._curnode, self._path)

    cpdef void update_root(self, object gs, Mutation m):
        """
        Should be called at the end of every bridge burning phase
        Changes _root to the new root
        """
        cdef ENode c
        for c in self._root._children:
            if c.m == m:
                self._root = c
                return

    cpdef void _add_root_noise(self):
        cdef int num_valid_moves = len(self._root._children)
        cdef float[:] noise = np.array(np.random.dirichlet(
            [NOISE_ALPHA_RATIO / (num_valid_moves+1)] * num_valid_moves
        ), dtype=np.float32)
        cdef ENode c
        cdef float n

        for n, c in zip(noise, self._root._children):
            c.p = c.p * (1 - self.root_noise_frac) + self.root_noise_frac * n
    
    cpdef void search(self, object gs, object nn, int sims, bint add_root_noise, bint add_root_temp):
        cdef float[:] v
        cdef float[:] p
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            p, v = nn(leaf.observation())
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)

    cpdef void raw_search(self, object gs, int sims, bint add_root_noise, bint add_root_temp):
        cdef Py_ssize_t policy_size = gs.action_size()
        cdef float[:] v = np.zeros(gs.num_players() + 1, dtype=np.float32)  #np.full((value_size,), 1 / value_size, dtype=np.float32)
        cdef float[:] p = np.full(policy_size, 1, dtype=np.float32)
        self.max_depth = 0

        for _ in range(sims):
            leaf = self.find_leaf(gs)
            self.process_results(leaf, v, p, add_root_noise, add_root_temp)
     
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
        leaf = gs.clone()

        # If the root sequence is not initialized
        if len(self._root.seq) < self.seq_length and not self._root.e.any():
            for a in self._root.seq:
                leaf.play_action(a)
            self.action_history = self._curnode.seq[0:gs.d]
        # Else if the root sequence is initialized, look for mutations
        else:
            while self._curnode.n > 0 and not self._curnode.e.any():
                self._path.append(self._curnode)
                #assert len(self._curnode._children) > 0, self._curnode
                self._curnode = self._curnode.best_child(self.fpu_reduction, self.cpuct)
                self.depth += 1

            if self.depth > self.max_depth:
                self.max_depth = self.depth
                self._discount_max_depth = self.depth

            if self._curnode.n == 0:
                valids = self.mutate_and_play(state = leaf,
                                            node = self._curnode,
                                        prior = self.mutate_prior)
                self.action_history = self._curnode.seq[0:gs.d]
                self._curnode.add_children( valid_mutations = valids, 
                                            mutate_prior = self.mutate_prior,
                                            num_players = self._num_players,
                                            player_history = self.player_history)
                # Playing the sequence
                self._curnode.e = leaf.win_state()

        return(leaf)

        
    cpdef np.ndarray mutate_and_play(self, object state, object node, np.ndarray prior):
        '''
        Returns valid_mutations
        '''
        cdef np.ndarray valids = np.zeros((self.seq_length, state.action_size()), dtype=np.float32)

        for t in range(0, self.seq_length):
            if state.win_state().any(): 
                break
            if t == node.m.tau:
                try:
                    state.play_action(node.m.a)
                    node.seq[t] = (node.m.a)
                except:
                    try:
                        state.play_action(node.seq[t])  
                    except:
                        a = np.argmax(prior[t]*state.valid_moves()).item()
                        state.play_action(a)  
                        node.seq[t] = a
            else:
                try:
                    state.play_action(node.seq[t])  
                except:
                    a = np.argmax(prior[t]*state.valid_moves()).item()
                    state.play_action(a)  
                    node.seq[t] = a

            valids[t] = state.valid_moves()
            valids[t, node.seq[t]] = 0  

        return valids

    cpdef void process_results_from_init(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        """
        Should be called at very beginning of the search, when the root sequence is not completed
        pi should help store the self.mutate_prior, which will be shared for all nodes.
        """
        if self.mutate_prior is None:
            self.mutate_prior = np.zeros((self.seq_length, gs.action_size()), dtype=np.float32)
        
        if self.policy_history is None:
            self.policy_history = np.zeros((self.seq_length, gs.action_size()), dtype=np.float32)
        
        assert self._curnode == self._root

        tau = len(self._root.seq)
        pi *= np.asarray(gs.valid_moves(), dtype=np.float32)
        # add root temperature
        if add_root_temp:
            pi = np.asarray(pi) ** (1.0 / self.root_temp)
            # re-normalize
            pi /= np.sum(pi)
                
        a = np.random.choice(len(pi), p = pi/np.sum(pi))

        self.policy_history[tau,a] += 1
        self.mutate_prior[tau] = pi
        self._root.seq.append(a)
        self.player_history.append(gs._player)

        player = gs._player
        gs.play_action(a)

        if gs._player!= player:
            # Pad the sequences at the of a turn so that a turn is always of length d
            idx, streak  = len(self._root.seq) -1, 1
            while idx>0 and self.player_history[idx-1] == player:
                idx -= 1
                streak += 1
            self.pad_histories(gs.d - streak, player)
        self._root.e = gs.win_state()

    cpdef void pad_histories(self, int pad, int player):
        self.player_history += [player for _ in range(pad)]
        self._root.seq += [-1 for _ in range(pad)]


    cpdef void process_results_from_mutations(self, object gs, float[:] value, float[:] pi, bint add_root_noise, bint add_root_temp):
        """
        Should be called when backpropagating along mutations. value, pi refers to the NN-evaluation of the state after playing the partial root_sequence
        """

        cdef Py_ssize_t num_players = gs.num_players()
        cdef ENode parent
        cdef float v
        cdef float discount
        cdef int i = 0

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





