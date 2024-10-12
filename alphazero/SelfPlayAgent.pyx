# cython: language_level=3

import torch.multiprocessing as mp
import numpy as np
import torch
import traceback
import itertools
import time

from alphazero.MCTS import MCTS


class SelfPlayAgent(mp.Process):
    def __init__(self, id, game_cls, ready_queue, batch_ready, batch_tensor, policy_tensor,
                 value_tensor, output_queue, result_queue, complete_count, games_played,
                 stop_event: mp.Event, pause_event: mp.Event(), args, _is_arena=False, _is_warmup=False):
        super().__init__()
        self.id = id
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
            self.mcts.append(self._get_mcts())

    def _get_mcts(self):
        if self._is_arena:
            return tuple([MCTS(self.args) for _ in range(self.game_cls.num_players())])
        else:
            return MCTS(self.args)

    def _mcts(self, index: int) -> MCTS:
        mcts = self.mcts[index]
        if self._is_arena:
            return mcts[self.games[index].player]
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
                self.fast = np.random.random_sample() < self.args.probFastSim
                sims = self.args.numFastSims if self.fast else self.args.numMCTSSims \
                    if not self._is_warmup else self.args.numWarmupSims
                
                """
                    for _ in range(sims):
                        if self.stop_event.is_set(): break
                        self.generateBatch()
                        if self.stop_event.is_set(): break
                        self.processBatch()
                    if self.stop_event.is_set(): break
                    self.playMoves()  
                """
                
                # VANILLA-MCTS
                if self.args.self_play_search_strategy == "VANILLA-MCTS":
                    for _ in range(sims):
                        if self.stop_event.is_set(): break
                        self.generateBatch()
                        if self.stop_event.is_set(): break
                        self.processBatch()
                    if self.stop_event.is_set(): break
                    self.updateTurn(mode = "complete", 
                                    transition_mode = "step") 
                    self.LogTurnToHistory()
                    self.processGameEnded()

                # BB-MCTS
                elif self.args.self_play_search_strategy == "BB-MCTS":
                    for _ in range(round(sims/self.game_cls.avg_atomic_actions())):
                        if self.stop_event.is_set(): break
                        self.generateBatch()
                        if self.stop_event.is_set(): break
                        self.processBatch()
                    if self.stop_event.is_set(): break
                    self.updateTurn(mode = "atomic", 
                                    transition_mode = "step") # extend the turn
                    self.LogTurnToHistory()
                    self.processGameEnded()

                # EMCTS
                elif self.args.self_play_search_strategy == "EMCTS":
                    for _ in range(sims):
                        if self.stop_event.is_set(): break
                        self.generateBatch()
                        if self.stop_event.is_set(): break
                        self.processBatch()
                    if self.stop_event.is_set(): break
                    self.updateTurn(mode = "complete", 
                                    transition_mode = "mutation") # Perform all the mutations
                    self.LogTurnToHistory()
                    self.playTurns() # if all the mutations ar econsumed then play full turn 
                    self.processGameEnded()

                # BB-EMCTS
                elif self.args.self_play_search_strategy == "BB-EMCTS":
                    for _ in range(round(sims/self.args.emcts_bb_phases)):
                        if self.stop_event.is_set(): break
                        self.generateBatch()
                        if self.stop_event.is_set(): break
                        self.processBatch()
                    if self.stop_event.is_set(): break
                    self.updateTurn(mode = "atomic", 
                                    transition_mode = "mutation") # mutate the turn once
                    self.LogTurnToHistory()
                    self.playTurns() # if all the mutations ar econsumed then play full turn 
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
            batch_tensor = [[] for _ in range(self.game_cls.num_players())]
            self.batch_indices = [[] for _ in range(self.game_cls.num_players())]

        for i in range(self.batch_size):
            self._check_pause()
            state = self._mcts(i).find_leaf(self.games[i])
            if self._is_warmup:
                self.policy_tensor[i].copy_(self._WARMUP_POLICY)
                self.value_tensor[i].copy_(self._WARMUP_VALUE)
                continue

            data = torch.from_numpy(state.observation())
            if self._is_arena:
                data = data.view(-1, *state.observation_size())
                player = self.player_to_index[self.games[i].player]
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
            self.ready_queue.put(self.id)

    def processBatch(self):
        if not self._is_warmup:
            self.batch_ready.wait()
            self.batch_ready.clear()

        for i in range(self.batch_size):
            self._check_pause()
            index = self.batch_indices[i] if self._is_arena else i
            self._mcts(i).process_results(
                self.games[i],
                self.value_tensor[index].data.numpy(),
                self.policy_tensor[index].data.numpy(),
                False if self._is_arena else self.args.add_root_noise,
                False if self._is_arena else self.args.add_root_temp
            )

    def updateTurn(self,
                    mode = "atomic",  # atomic: extend the turn, complete: complete the turn
                    transition_mode = "step" # step: construct the turn 1 by 1 or mutation: mutate the turn
                    ):

        # BB-MCTS
        if mode == "atomic" and transition_mode == "step":
            for i in range(self.batch_size):
                
                self._check_pause()
                self.temps[i] = self.args.temp_scaling_fn(
                    self.temps[i], self.games[i].turns, self.game_cls.max_turns()
                ) if not self._is_arena else self.args.arenaTemp

                self._mcts(i).update_turn(self.games[i], self.temps[i])

        # VANILLA-MCTS 
        elif mode == "complete" and transition_mode == "step":
            for i in range(self.batch_size):
                self._check_pause()
                self.temps[i] = self.args.temp_scaling_fn(
                    self.temps[i], self.games[i].turns, self.game_cls.max_turns()
                ) if not self._is_arena else self.args.arenaTemp

                while not self._mcts(i).turn_completed:
                    self._mcts(i).update_turn(self.games[i], self.temps[i])

        # EMCTS
        elif mode == "complete" and transition_mode == "mutate":          
            for i in range(self.batch_size):

                self._check_pause()
                self.temps[i] = self.args.temp_scaling_fn(
                    self.temps[i], self.games[i].turns, self.game_cls.max_turns()
                ) if not self._is_arena else self.args.arenaTemp
                
                for _ in self.args.emcts_bb_phases: 
                    self.emcts(i).mutate_turn(self.games[i], self.temps[i])

        # BB- EMCTS
        elif mode == "complete" and transition_mode == "mutate":           
            for i in range(self.batch_size):
                if self.emcts(i).bb_phase == 0:
                        self.emcts(i).reset()
                        self.emcts(i).init_turn(self.games[i], self.temps[i])
                self._check_pause()
                self._mcts(i).reset()
                self.temps[i] = self.args.temp_scaling_fn(
                    self.temps[i], self.games[i].turns, self.game_cls.max_turns()
                ) if not self._is_arena else self.args.arenaTemp
                
                self.emcts(i).mutate_turn(self.games[i], self.temps[i])

                    
    def LogTurnToHistory(self):
        for i in range(self.batch_size):
            if self._mcts(i).turn_completed:
                for t in range(len(self._mcts(i).state_history)):
                    state = self._mcts(i).state_history[t]
                    pi = self._mcts(i).policy_history[t]
                    self.histories[i].append((state, pi))
                self._mcts(i).reset()
                # self.emcts(i).reset()
                # self.emcts(i).init_turn(self.games[i], self.temps[i])
                

    def processGameEnded(self):
        for i in range(self.batch_size):
            self._check_pause()
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
    def playMoves(self):
        for i in range(self.batch_size):
            self._check_pause()
            self.temps[i] = self.args.temp_scaling_fn(
                self.temps[i], self.games[i].turns, self.game_cls.max_turns()
            ) if not self._is_arena else self.args.arenaTemp
            
            if self.args.self_play_search_strategy == "BB-MCTS":
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
                

            if self.args.self_play_search_strategy == "VANILLA-MCTS":
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