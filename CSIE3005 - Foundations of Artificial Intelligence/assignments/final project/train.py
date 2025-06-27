# v7

from typing import Dict, Tuple

import random
import os
import time
import pickle

from game.engine.poker_constants import PokerConstants as Const
from game.engine.table import Table
from game.engine.round_manager import RoundManager
from game.engine.player import Player
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card


class InfoSet:
    ACTION_TYPE = ["FOLD", "CALL", "RAISE", "ALLIN"]

    HAND_STRENGTH_MAP = {
        "HIGHCARD": 1,
        "ONEPAIR": 2,
        "TWOPAIR": 3,
        "THREECARD": 4,
        "STRAIGHT": 5,
        "FLASH": 6,
        "FULLHOUSE": 7,
        "FOURCARD": 8,
        "STRAIGHTFLASH": 9,
    }

    def __init__(self):
        # dict[info_key_l1][info_key_l2][action] = value
        self.regret_sum: Dict[Tuple, Dict[Tuple, Dict[str, float]]] = {}
        self.strategy_sum: Dict[Tuple, Dict[Tuple, Dict[str, float]]] = {}
        self.avg_strategy: Dict[Tuple, Dict[Tuple, Dict[str, float]]] = {}

    def build_info_key(self, hole_card, round_state, current_player) -> Tuple[Tuple, Tuple]:
        # tuple(street, hole_highest_rank, community_highest_rank, hand_strength, hand_rank), action_summary
        
        street = round_state['street']

        # We don't store cards directly, since there's too much combinations.
        # Instead, we store the strength and rank.
        hole = [Card.from_str(c) for c in hole_card]
        community = [Card.from_str(c) for c in round_state['community_card']]
        hole_highest_rank = self.__find_highest_rank(hole)
        community_highest_rank = self.__find_highest_rank(community)
        hand_strength, hand_rank = self.__calculate_hand_strength(hole, community)

        # We store opponent's action history, since opponent's actions may indicate their hand strength.
        history = [action for street_history in round_state['action_histories'].values() for action in street_history if action['uuid'] != current_player]
        action_summary = [0, 0, 0, 0] # (CALLHIGH, CALLLOW, RAISEHIGH, RAISELOW)
        for action in history:
            if action['action'] == 'CALL' and action['amount'] > 500:
                action_summary[0] += 1
            elif action['action'] == 'CALL' and action['amount'] <= 500:
                action_summary[1] += 1
            elif action['action'] == 'RAISE' and action['amount'] > 500:
                action_summary[2] += 1
            elif action['action'] == 'RAISE' and action['amount'] <= 500:
                action_summary[3] += 1

        return (street, hole_highest_rank, community_highest_rank, hand_strength, hand_rank), tuple(action_summary)

    def __calculate_hand_strength(self, hole_card, community_card) -> Tuple[int, int]:
        if len(community_card) < 3:
            rank0 = hole_card[0].rank
            rank1 = hole_card[1].rank

            if rank0 == rank1:
                return 2, rank0 # pair
            else:
                return 1, max(rank0, rank1) # high card

        else:
            strength = HandEvaluator.gen_hand_rank_info(hole_card, community_card)
            return self.HAND_STRENGTH_MAP[strength["hand"]["strength"]], strength["hand"]["rank_1"]
        
    def __find_highest_rank(self, cards) -> int:
        if not cards:
            return 0
        return max(card.rank for card in cards)
    
    def get_strategy_from_regret(self, info_key: Tuple, epsilon = 0.05) -> Dict[str, float]:
        l1, l2 = info_key

        if l1 not in self.regret_sum:
            return self.__uniform_strategy()

        if l2 not in self.regret_sum[l1]:
            return self.__l1_average_strategy_from_regret(l1)

        regrets = self.regret_sum[l1][l2]
        positive_regrets = {a: max(0.0, regrets.get(a, 0.0)) for a in regrets}
        total = sum(positive_regrets.values())

        if total > 0: 
            strategy = {a: positive_regrets.get(a, 0.0) / total for a in self.ACTION_TYPE}
        else:
            strategy = self.__l1_average_strategy_from_regret(l1)

        strategy = {
            a: (1 - epsilon) * strategy.get(a, 0.0) + epsilon / len(self.ACTION_TYPE)
            for a in strategy
        }
        return strategy

    def __uniform_strategy(self) -> Dict[str, float]:
        n = len(self.ACTION_TYPE)
        return {a: 1/n for a in self.ACTION_TYPE}

    def __l1_average_strategy_from_regret(self, l1) -> Dict[str, float]:
        # when l1 is defined, but can't find corresponding l2 (unseen action_summary)
        # sum up all strategies for l1
        total_regrets = {a: 0.0 for a in self.ACTION_TYPE}
        for l2 in self.regret_sum[l1]:
            regrets = self.regret_sum[l1][l2]
            for action in self.ACTION_TYPE:
                total_regrets[action] += regrets.get(action, 0.0)

        positive_regrets = {a: max(0.0, total_regrets.get(a, 0.0)) for a in self.ACTION_TYPE}
        total = sum(positive_regrets.values())
        
        if total > 0:
            return {a: positive_regrets.get(a, 0.0) / total for a in self.ACTION_TYPE}
        else:
            return self.__uniform_strategy()
        
    def get_average_strategy(self, info_key: Tuple) -> Dict[str, float]:
        l1, l2 = info_key

        if l1 not in self.avg_strategy:
            return self.__uniform_strategy()

        if l2 not in self.avg_strategy[l1]:
            return self.__l1_average_strategy(l1)

        return self.avg_strategy[l1][l2]

    def __l1_average_strategy(self, l1) -> Dict[str, float]:
        # when l1 is defined, but can't find corresponding l2 (unseen action_summary)
        # sum up all strategies for l1
        total_strategy = {a: 0.0 for a in self.ACTION_TYPE}
        for l2 in self.strategy_sum[l1]:
            strategy = self.strategy_sum[l1][l2]
            for action in self.ACTION_TYPE:
                total_strategy[action] += strategy.get(action, 0.0)

        total = sum(total_strategy.values())
        if total > 0:
            return {a: total_strategy.get(a, 0.0) / total for a in self.ACTION_TYPE}
        else:
            return self.__uniform_strategy()

    def __generate_average_strategy(self):
        for l1 in self.strategy_sum:
            self.avg_strategy[l1] = {}
            for l2 in self.strategy_sum[l1]:
                strategy_sum = self.strategy_sum[l1][l2]
                total = sum(strategy_sum.values())

                if total > 0:
                    strategy = {
                        action: strategy_sum.get(action, 0.0) / total
                        for action in self.ACTION_TYPE
                    }
                else:
                    strategy = self.__uniform_strategy()

                self.avg_strategy[l1][l2] = self.__normalize_strategy(strategy)

    def __normalize_strategy(self, strategy) -> Dict[str, float]:
        # add small epsilon to avoid zero probabilities
        epsilon = 0.05
        smoothed = {a: strategy[a] + epsilon for a in self.ACTION_TYPE}
        total_smoothed = sum(smoothed.values())
        normalized = {a: smoothed[a] / total_smoothed for a in self.ACTION_TYPE}
        return normalized

    def normalize_sums(self):
        for l1 in self.regret_sum:
            for l2 in self.regret_sum[l1]:
                norm = sum(abs(v) for v in self.regret_sum[l1][l2].values())
                if norm > 1e-6:
                    for a in self.regret_sum[l1][l2]:
                        self.regret_sum[l1][l2][a] /= norm

    def update_regret_sum(self, info_key, action, regret):
        l1, l2 = info_key

        if l1 not in self.regret_sum:
            self.regret_sum[l1] = {}
        if l2 not in self.regret_sum[l1]:
            self.regret_sum[l1][l2] = {a: 0.0 for a in self.ACTION_TYPE}
        if action not in self.regret_sum[l1][l2]:
            raise ValueError(f"action {action} is not a valid action type.")

        self.regret_sum[l1][l2][action] += regret

    def update_strategy_sum(self, info_key, action, prob):
        l1, l2 = info_key

        if l1 not in self.strategy_sum:
            self.strategy_sum[l1] = {}
        if l2 not in self.strategy_sum[l1]:
            self.strategy_sum[l1][l2] = {a: 0.0 for a in self.ACTION_TYPE}
        if action not in self.strategy_sum[l1][l2]:
            raise ValueError(f"action {action} is not a valid action type.")

        self.strategy_sum[l1][l2][action] += prob
            
    def save_train(self, filename):
        # Save the regret_sum and strategy_sum to a file
        with open(filename, 'wb') as f:
            if isinstance(self.regret_sum, dict) and isinstance(self.strategy_sum, dict):
                pickle.dump((self.regret_sum, self.strategy_sum), f)
            else:
                raise ValueError("regret_sum and strategy_sum must be dictionaries.")

    def load_train(self, filename):
        # Load the regret_sum and strategy_sum from a file
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.regret_sum, self.strategy_sum = data
            if not isinstance(self.regret_sum, dict) or not isinstance(self.strategy_sum, dict):
                raise ValueError("Loaded data must be dictionaries for regret_sum and strategy_sum.")

    def save(self, filename):
        # Save the average strategy to a file
        self.__generate_average_strategy()
        with open(filename, 'wb') as f:
            pickle.dump(self.avg_strategy, f)
    
    def load(self, filename):
        # Load the average strategy from a file
        with open(filename, 'rb') as f:
            self.avg_strategy = pickle.load(f)


class ActionBuilder:
    @staticmethod
    def get_possible_actions(valid_actions) -> Dict[str, Tuple[str, int]]:
        """
        List all possible actions based on the valid actions.
        """
        possible_actions = {}
        for action in valid_actions:
            if action["action"] == "fold":
                possible_actions["FOLD"] = ("fold", 0)
            elif action["action"] == "call":
                possible_actions["CALL"] = ("call", action["amount"])
            elif action["action"] == "raise":
                possible_actions["RAISE"] = ("raise", action["amount"]["min"])
                possible_actions["ALLIN"] = ("raise", action["amount"]["max"])

        return possible_actions

    @staticmethod
    def get_strategy_action(strategy: Dict[str, float], possible_actions: Dict[str, Tuple[str, int]]) -> Tuple[str, int]:
        """
        Choose an action based on the strategy.
        """
        valid_act = list(set(strategy.keys()) & set(possible_actions.keys()))
        probabilities = [strategy[a] for a in valid_act]
        act = random.choices(valid_act, weights=probabilities)[0]
        return possible_actions[act]
    
    @staticmethod
    def get_sample_action(possible_actions: Dict[str, Tuple[str, int]]) -> Tuple[str, int]:
        """
        Randomly select an action from the possible actions.
        """
        return random.choice(list(possible_actions.values()))


class MCCFRTrainer:
    def __init__(self):
        self.info_set = InfoSet()
        self.sb_amount = 5
        self.ante = 0
        self.initial_stack = 1000

        # round specific
        self.round_table = Table()
        self.round_updating_player = ""

    def train(self, iterations = 1, save_interval = 1000, trainfile = None, prefix = ""):
        total_time = time.time()

        if trainfile and os.path.exists(trainfile):
            print("previous trainfile found in the current directory.")
            trainer.info_set.load_train(trainfile)
        
        for i in range(iterations):
            start_time = time.time()
            self.start_new_round()
            print(f"round {i+1}/{iterations} finished in {time.time() - start_time:.2f} seconds")

            if save_interval and (i + 1) % save_interval == 0 and i < iterations - 10:
                self.info_set.normalize_sums()

                print(f"saving progress at iteration {i+1}...")
                self.info_set.save_train(f"{prefix}.temp.train.pkl")
                print("progress saved.")

        if os.path.exists(f"{prefix}.temp.train.pkl"):
            os.remove(f"{prefix}.temp.train.pkl")

        self.info_set.save_train(f"{prefix}.train.pkl")
        self.info_set.save(f"{prefix}.pkl")
        print(f"training data saved to {prefix}.train.pkl and {prefix}.pkl")
        print(f"training finished in {time.time() - total_time:.2f} seconds")

    def start_new_round(self):
        self.round_table = Table()

        self.__add_player("p0")
        self.__add_player("p1")
        self.round_table.set_blind_pos(0, 1)

        init_state, init_msgs = RoundManager.start_new_round(
            1, self.sb_amount, self.ante, self.round_table
        )

        self.round_updating_player = init_state["table"].seats.players[0].uuid
        self.traverse(init_state, init_msgs, 1, 0)

    ############################################################################
    # MCCFR ALGORITHM FUNCTIONS
    ############################################################################
    def traverse(self, state, msgs, reach_prop, depth) -> float:
        current_player, msg = msgs[-1]
        
        if self.__is_terminal(state, msg):
            return self.__get_payoff(state)


        valid_actions, hole_card, round_state = self.__parse_ask_message(msg["message"])

        info_key = self.info_set.build_info_key(hole_card, round_state, current_player)
        strategy = self.info_set.get_strategy_from_regret(info_key, max(0.01, 0.1 * (0.99 ** depth)))

        possible_actions = ActionBuilder.get_possible_actions(valid_actions)

        if current_player == self.round_updating_player:
            util = {}
            node_util = 0
            
            # external sampling: traverse all actions
            for act_type, action in possible_actions.items():
                next_state, next_msgs = RoundManager.apply_action(state, action[0], action[1])
                
                util[act_type] = self.traverse(next_state, next_msgs, reach_prop * strategy[act_type], depth + 1) 
                node_util += strategy[act_type] * util[act_type]

            for act_type in possible_actions.keys():
                regret = util[act_type] - node_util
                self.info_set.update_regret_sum(info_key, act_type, regret)
                self.info_set.update_strategy_sum(info_key, act_type, reach_prop * strategy[act_type])

            return node_util
        
        else:
            act, amt = ActionBuilder.get_sample_action(possible_actions)
            next_state, next_msgs = RoundManager.apply_action(state, act, amt)

            for act_type in possible_actions.keys():
                self.info_set.update_strategy_sum(info_key, act_type, reach_prop * strategy[act_type])
            
            return self.traverse(next_state, next_msgs, reach_prop, depth + 1)

    def __is_terminal(self, state, msg):
        street = state['street']
        if (street == Const.Street.FINISHED or msg["message"]["message_type"] == "round_result_message"):
            return True
        return False

    def __get_payoff(self, state):
        table = state["table"]
        final_stack = next(p.stack for p in table.seats.players if p.uuid == self.round_updating_player)
        return final_stack - self.initial_stack

    ############################################################################
    # GAME LOGIC FUNCTIONS
    ############################################################################
    def __add_player(self, player_name):
        uuid = self.__generate_uuid()
        player = Player(uuid, self.initial_stack, player_name)
        self.round_table.seats.sitdown(player)

    def __generate_uuid(self):
        uuid_size = 22
        chars = [chr(code) for code in range(97, 123)]
        return "".join([random.choice(chars) for _ in range(uuid_size)])
    
    def __parse_ask_message(self, message):
        valid_actions = message["valid_actions"]
        hole_card = message["hole_card"]
        round_state = message["round_state"]
        return valid_actions, hole_card, round_state
    

# nohup python -u train.py &> train.log &
if __name__ == "__main__":
    trainer = MCCFRTrainer()
    trainer.train(iterations=300000, save_interval=50000, trainfile="v7_200k.train.pkl", prefix="v7_500k")
