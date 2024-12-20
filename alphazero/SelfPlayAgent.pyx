# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools
import time

from alphazero.MCTS import MCTS
from alphazero.EMCTS import EMCTS

class SelfPlayAgent(mp.Process):

    """
    This process has two goals:
        - Allow self-play matches and logging the decisions to the output_queue with a given tree search self-play strategy
        - (_is_arena) Allow arena matches, which should support players with different tree search strategies, specified in arena_configutations

    """
    def __init__(self, id_process, game_cls, arena_configurations, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, pause_event: mp.Event(), args, _is_arena=False, _is_warmup=False):
        super().__init__()
        self.arena_configurations = arena_configurations # [(mode_p1, strategy_p1, args1), (mode_p2, strategy_p2, args2), ...] if arena else None
        self.id_process = id_process
        self.game_cls = game_cls
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        if _is_arena:
            self.batch_size = policy_tensor.shape[0]
        else:
            self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []
        self.histories = []
        self.temps = []
        self.next_reset = []
        self.sims = []
        self.gplayers = []
        self.mcts = []
        self.games_played = games_played
        self.complete_count = complete_count
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.args = args

        self._is_arena = _is_arena
        self._is_warmup = _is_warmup
        if _is_arena:
            self.player_to_index = list(range(game_cls.num_players()))
            np.random.shuffle(self.player_to_index)
            self.batch_indices = None
        if _is_warmup:
            action_size = game_cls.action_size()
            self._WARMUP_POLICY = torch.full((action_size,), 1 / action_size).to(policy_tensor.device)
            value_size = game_cls.num_players() + 1
            self._WARMUP_VALUE = torch.full((value_size,), 1 / value_size).to(policy_tensor.device)

        self.fast = False
        for _ in range(self.batch_size):
            self.games.append(self.game_cls())
            self.histories.append([])
            self.temps.append(self.args.startTemp)
            self.next_reset.append(0)
            self.sims.append(0)
            self.gplayers.append(0) # represents who is playing on is currently playing on the game i
            self.mcts.append(self._get_mcts())
    
    def _get_mode_and_strategy(self, i: int):
        if self._is_arena:
            mode, strategy, args = self.arena_configurations[self.gplayers[i]]
        else:
            mode, strategy = self.args.self_play_mode, self.args.self_play_strategy
        return(mode, strategy)

    def _get_mcts(self):
        if not self._is_arena:
            if self.args.self_play_mode == "mcts":
                return MCTS(self.args)
            elif self.args.self_play_mode == "emcts":
                return EMCTS(self.args)
            else:
                raise ValueError

        elif self._is_arena:
            mcts = []
            for player in range(self.game_cls.num_players()):
                mode = self.arena_configurations[player][0]
                if mode =='mcts':
                    args = self.arena_configurations[player][2]
                    mcts.append(MCTS(args))
                elif mode =='emcts':
                    args = self.arena_configurations[player][2]
                    mcts.append(EMCTS(args))
                else:
                    raise ValueError, f"Mode {mode} not in ['mcts','emcts']"
            return tuple(mcts)

    def _mcts(self, index: int) -> MCTS:
        mcts = self.mcts[index]
        if self._is_arena:
            return mcts[self.gplayers[index]]
        else:
            return mcts

    def _check_pause(self):
        while self.pause_event.is_set():
            time.sleep(.1)

    def run(self):
        try:
            np.random.seed()
            while not self.stop_event.is_set() and self.games_played.value < self.args.gamesPerIteration:
                self._check_pause()

                if self.stop_event.is_set(): break
                self.generateBatch() # call find_leaf (selection and expansion), and create the batch of observations that should be processed by NN
                if self.stop_event.is_set(): break
                self.processBatch() # call process_results (backpropagation)
                if self.stop_event.is_set(): break
                self.updateTurn() # after a certain number of sims, update the turn: extend the sequence (mcts), or mutate the sequence (emcts), and reset if the turn is completed
                if self.stop_event.is_set(): break
                self.processGameEnded()
    
            with self.complete_count.get_lock():
                self.complete_count.value += 1
            if not self._is_arena:
                self.output_queue.close()
                self.output_queue.join_thread()
        except Exception:
            print(traceback.format_exc())

    def generateBatch(self):
        if self._is_arena:
            batch_tensor = [[] for _ in range(self.game_cls.num_players())] # BATCH OF GAMES TO EVALUATE
            self.batch_indices = [[] for _ in range(self.game_cls.num_players())] # INDICES OF THE GAMES 

        for i in range(self.batch_size):
            self._check_pause()
            state = self._mcts(i).find_leaf(self.games[i])
            self.sims[i]+=1
            if self._is_warmup:
                self.policy_tensor[i].copy_(self._WARMUP_POLICY)
                self.value_tensor[i].copy_(self._WARMUP_VALUE)
                continue

            data = torch.from_numpy(state.observation())
            if self._is_arena:
                data = data.view(-1, *state.observation_size())
                player = self.player_to_index[self.gplayers[i]]
                batch_tensor[player].append(data)
                self.batch_indices[player].append(i)
            else:
                self.batch_tensor[i].copy_(data)

        if self._is_arena:
            for player in range(self.game_cls.num_players()):
                player = self.player_to_index[player]
                data = batch_tensor[player]
                if data:
                    batch_tensor[player] = torch.cat(data)
            self.output_queue.put(batch_tensor)
            self.batch_indices = list(itertools.chain.from_iterable(self.batch_indices))
        if not self._is_warmup:
            self.ready_queue.put(self.id_process)

    def processBatch(self):
        if not self._is_warmup:
            self.batch_ready.wait()
            self.batch_ready.clear()

        for i in range(self.batch_size):
            self._check_pause()

            mode, strategy = self._get_mode_and_strategy(i)
            index = self.batch_indices[i] if self._is_arena else i

            if mode =='emcts': 
                leaf = self._mcts(i).reconstruct_leaf(self.games[i])
                self._mcts(i).process_results(
                    leaf,
                    self.value_tensor[index].data.numpy(),
                    self.policy_tensor[index].data.numpy(),
                    False if self._is_arena else self.args.add_root_noise,
                    False if self._is_arena else self.args.add_root_temp
                )
            else:
                self._mcts(i).process_results(
                    self.games[i],
                    self.value_tensor[index].data.numpy(),
                    self.policy_tensor[index].data.numpy(),
                    False if self._is_arena else self.args.add_root_noise,
                    False if self._is_arena else self.args.add_root_temp
                )

    def _sims_threshold(self, mode, strategy, game):
        if mode == "mcts" and strategy == "vanilla":
            return(self.args.numMCTSSims)
        elif mode== "mcts" and strategy == "bridge-burning":
            return(self.args.numMCTSSims/ game.avg_atomic_actions())
        elif mode == "emcts" and strategy == "vanilla":
            return(self.args.numMCTSSims)
        elif mode== "emcts" and strategy == "bridge-burning":
            return(self.args.numMCTSSims/ self.args.emcts_bb_phases)
        else:
            raise ValueError, f"Strategy {strategy} not in ['bridge-burning', 'vanilla'] or Node {mode} not in ['mcts','emcts']"

    def updateTurn(self):
        for i in range(self.batch_size):
            self._check_pause()
            mode, strategy = self._get_mode_and_strategy(i)
            if self.sims[i]>= self._sims_threshold(mode, strategy, self.games[i]):

                self.temps[i] = self.args.temp_scaling_fn(
                        self.temps[i], self.games[i].turns, self.game_cls.max_turns()
                    ) if not self._is_arena else self.args.arenaTemp
                
                if strategy == 'vanilla':
                    while not self._mcts(i).turn_completed:
                        self._mcts(i).update_turn(self.games[i], self.temps[i])

                elif strategy == 'bridge-burning':
                    if not self._mcts(i).turn_completed: 
                        self._mcts(i).update_turn(self.games[i], self.temps[i])
                    
            ## If the turn is completed, then log the state history and policy history to the retrain buffer, and reset
            if self._mcts(i).turn_completed:
                turn, pi, state_history = self._mcts(i).get_results(self.games[i])
                for t in range(len(turn)):
                    self.histories[i].append((state_history[t], pi[t]))
                self._mcts(i).reset()
                self.sims[i] = 0
                self.gplayers[i] = (self.gplayers[i] + 1) % self.game_cls.num_players()
                
    def processGameEnded(self):
        for i in range(self.batch_size):
            self._check_pause()
            winstate = self.games[i].win_state()
            if winstate.any():
                self.result_queue.put((self.games[i].clone(), winstate, self.id_process))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            self._check_pause()
                            if self.args.symmetricSamples:
                                data = hist[0].symmetries(hist[1])
                            else:
                                data = ((hist[0], hist[1]),)

                            for state, pi in data:
                                self._check_pause()
                                self.output_queue.put((
                                    state.observation(), pi, np.array(winstate, dtype=np.float32)
                                ))
                    self.games[i] = self.game_cls()
                    self.histories[i] = []
                    self.temps[i] = self.args.startTemp
                    self.mcts[i] = self._get_mcts()
                else:
                    lock.release()

    """
    def playMoves(self):
        for i in range(self.batch_size):
            self._check_pause()
            self.temps[i] = self.args.temp_scaling_fn(
                self.temps[i], self.games[i].turns, self.game_cls.max_turns()
            ) if not self._is_arena else self.args.arenaTemp
            
            if self.args.self_play_strategy == "BB-MCTS":
                policy = self._mcts(i).probs(self.games[i], self.temps[i])
                action = np.random.choice(self.games[i].action_size(), p=policy)
                if not self.fast and not self._is_arena:
                    self.histories[i].append((
                        self.games[i].clone(),
                        self._mcts(i).probs(self.games[i])
                    ))

                if self._is_arena:
                    [mcts.update_root(self.games[i], action) for mcts in self.mcts[i]]
                else:
                    self._mcts(i).update_root(self.games[i], action)
                self.games[i].play_action(action)
                

            if self.args.self_play_strategy == "VANILLA-MCTS":
                current_player = self.games[i].clone()._player
                while  self.games[i]._player == current_player and not self.games[i].win_state().any():
                    self._mcts(i).raw_search(self.games[i], 1, 0, 0)
                    policy = self._mcts(i).probs(self.games[i], self.temps[i])
                    action = np.random.choice(self.games[i].action_size(), p=policy)
                    if not self.fast and not self._is_arena:
                        self.histories[i].append((
                            self.games[i].clone(),
                            self._mcts(i).probs(self.games[i])
                        ))
                    if self._is_arena:
                        [mcts.update_root(self.games[i], action) for mcts in self.mcts[i]]
                    else:
                        self._mcts(i).update_root(self.games[i], action)
                    self.games[i].play_action(action)
                

            if self.args.mctsResetThreshold and self.games[i].turns >= self.next_reset[i]:
                self.mcts[i] = self._get_mcts()
                self.next_reset[i] = self.games[i].turns + self.args.mctsResetThreshold

            winstate = self.games[i].win_state()
            if winstate.any():
                self.result_queue.put((self.games[i].clone(), winstate, self.id))
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.gamesPerIteration:
                    self.games_played.value += 1
                    lock.release()
                    if not self._is_arena:
                        for hist in self.histories[i]:
                            self._check_pause()
                            if self.args.symmetricSamples:
                                data = hist[0].symmetries(hist[1])
                            else:
                                data = ((hist[0], hist[1]),)

                            for state, pi in data:
                                self._check_pause()
                                self.output_queue.put((
                                    state.observation(), pi, np.array(winstate, dtype=np.float32)
                                ))
                    self.games[i] = self.game_cls()
                    self.histories[i] = []
                    self.temps[i] = self.args.startTemp
                    self.mcts[i] = self._get_mcts()
                else:
                    lock.release()
    """