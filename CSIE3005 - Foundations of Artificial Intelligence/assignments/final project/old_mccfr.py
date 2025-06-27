from collections import defaultdict
from typing import List, Dict, Tuple
from enum import Enum

import random
import os
import time
import json

from game.engine.poker_constants import PokerConstants as Const
from game.engine.table import Table
from game.engine.round_manager import RoundManager
from game.engine.player import Player
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card


class ACTION_TYPE(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE_MIN = "raise_min"
    RAISE_HALF = "raise_half"
    RAISE_ONE = "raise_one"
    RAISE_DOUBLE = "raise_double"
    RAISE_MAX = "raise_max"


class DictHelper:
    @staticmethod
    def defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: DictHelper.defaultdict_to_dict(v) for k, v in d.items()}
        return d

    @staticmethod
    def dict_to_defaultdict(d):
        result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for k1, v1 in d.items():
            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    result[k1][k2][k3] = v3
        return result


class InfoSet:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.strategy_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.version = 0

    def build_info_key(self, hole_card, round_state, current_player) -> Tuple[str, str]:

        if self.version == 1:
            street = round_state['street']
            hole_str = '-'.join(hole_card)
            community_card = round_state['community_card']
            community_str = '-'.join(community_card) if community_card else ''
            pot_size = int(round_state['pot']['main']['amount'])
            pot_size_str = "HIGH" if pot_size > 500 else "LOW"
            l1 = f"{street},{hole_str},{community_str},{pot_size_str}"

            # build action history string
            history = round_state['action_histories'].get(street, [])
            action_count = defaultdict(int)
            for action in history:
                action_count[action['action']] += 1
            l2 = '-'.join([f"{action}:{action_count[action]}" for action in ["FOLD", "CALL", "RAISE"]])

            return (l1, l2)

        elif self.version == 2:
            street = round_state['street']

            community_card = round_state['community_card']
            hole_str = '-'.join(hole_card)
            community_str = '-'.join(community_card) if community_card else ''

            # get hand strength
            hand_strength, hand_rank, hole_rank = self.__calculate_hand_strength(hole_card, community_card)
            strength_str = f"{hand_strength},{hand_rank},{hole_rank}"

            # get opponent's action history
            history = [action for street_history in round_state['action_histories'].values() for action in street_history if action['uuid'] != current_player]
            history_str = ""
            for action in history:
                if action['action'] == 'CALL' or action['action'] == 'RAISE':
                    amount = "h" if action['amount'] > 500 else "l"
                    history_str += f"{action['action']}{amount}"

            l1 = f"{street},{hole_str},{community_str},{strength_str}"
            l2 = f"{history_str}"

            return (l1, l2)
        
        else:
            raise ValueError(f"Unsupported version: {self.version}")

    def __calculate_hand_strength(self, hole_card, community_card):
        """
        return (hand_strength, hand_rank, hole_rank)\n
        """
        if len(community_card) < 3:
            hole = [Card.from_str(c) for c in hole_card] 
            rank0 = hole[0].rank
            rank1 = hole[1].rank

            if rank0 == rank1:
                if rank0 > 10:
                    return 1, 1, rank0 # strong pair
                else:
                    return 1, 0, rank0
            else:
                return 0, 0, max(rank0, rank1) # high card

        else:
            hole = [Card.from_str(c) for c in hole_card]
            community = [Card.from_str(c) for c in community_card]
            strength = HandEvaluator.gen_hand_rank_info(hole, community)
            return strength["hand"]["strength"], strength["hand"]["high"], strength["hole"]["high"]

    def get_strategy(self, info_key: Tuple[str, str], epsilon = 0.05) -> Dict[str, float]:
        l1, l2 = info_key

        if l1 not in self.regret_sum:
            return self.__uniform_strategy()

        l1_regrets = self.regret_sum[l1]
        if l2 not in l1_regrets:
            return self.__l1_average_strategy(l1)

        regrets = l1_regrets[l2]
        positive_regrets = {a: max(0.0, regrets.get(a, 0.0)) for a in regrets}
        total = sum(positive_regrets.values())

        if total > 0: 
            strategy = {a.value: positive_regrets.get(a.value, 0.0) / total for a in ACTION_TYPE}
        else:
            strategy = self.__l1_average_strategy(l1)

        # strategy = {
        #     a: (1 - epsilon) * strategy.get(a, 0.0) + epsilon / len(ACTION_TYPE)
        #     for a in strategy
        # }
        return strategy

    def __uniform_strategy(self) -> Dict[str, float]:
        n = len(ACTION_TYPE)
        return {a.value: 1/n for a in ACTION_TYPE}

    def __l1_average_strategy(self, l1) -> Dict[str, float]:
        # when l1 is defined, but can't find corresponding l2 (unseen action history)
        # sum up all strategies for l1
        l1_regrets = self.regret_sum[l1]
        total_regrets = defaultdict(float)
        for l2 in l1_regrets:
            regrets = l1_regrets[l2]
            for action in ACTION_TYPE:
                total_regrets[action] += regrets.get(action, 0.0)

        positive_regrets = {a.value: max(0.0, total_regrets.get(a.value, 0.0)) for a in ACTION_TYPE}
        total = sum(positive_regrets.values())
        
        if total > 0:
            return {a.value: positive_regrets.get(a.value, 0.0) / total for a in ACTION_TYPE}
        else:
            return self.__uniform_strategy()

    def update_regret_sum(self, info_key, action, regret):
        l1, l2 = info_key
        self.regret_sum[l1][l2][action] += regret

    def update_strategy_sum(self, info_key, action, prob):
        l1, l2 = info_key
        self.strategy_sum[l1][l2][action] += prob
            
    def save(self, filename):
        # Save the regret_sum and strategy_sum to a file
        data = {
            "regret_sum": DictHelper.defaultdict_to_dict(self.regret_sum),
            "strategy_sum": DictHelper.defaultdict_to_dict(self.strategy_sum)
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filename):
        # Load the regret_sum and strategy_sum from a file
        with open(filename, 'r') as f:
            data = json.load(f)
            self.regret_sum = DictHelper.dict_to_defaultdict(data["regret_sum"])
            self.strategy_sum = DictHelper.dict_to_defaultdict(data["strategy_sum"])


class ActionBuilder:
    @staticmethod
    def __get_discretized_bets(min_raise, max_raise, pot_size) -> List[Tuple[str, int]]:
        """
        List all possible raise amounts based on the min and max raise values.\n
        return [(act_type, amt)]\n
        """
        abstract = [(ACTION_TYPE.RAISE_MIN.value, min_raise)]
        # if pot_size * 0.5 < max_raise:
        #     abstract.append((ACTION_TYPE.RAISE_HALF.value, pot_size * 0.5))
        if pot_size < max_raise:
            abstract.append((ACTION_TYPE.RAISE_ONE.value, max(pot_size, min_raise)))
        if pot_size * 2 < max_raise:
            abstract.append((ACTION_TYPE.RAISE_DOUBLE.value, max(pot_size * 2, min_raise)))
        if abstract[-1][1] < max_raise:
            abstract.append((ACTION_TYPE.RAISE_MAX.value, max_raise))
        return abstract
    
    @staticmethod
    def get_possible_actions(valid_actions, pot_size) -> List[Tuple[str, str, int]]:
        """
        List all possible actions based on the valid actions.\n
        return [(act_type, act, amt)]\n
        """
        actions = []
        for action in valid_actions:
            if action["action"] == "raise":
                min_raise = action["amount"]["min"]
                max_raise = action["amount"]["max"]
                abstract = ActionBuilder.__get_discretized_bets(min_raise, max_raise, pot_size)
                for act_type, amt in abstract:
                    actions.append((act_type, "raise", amt))
            else:
                actions.append((action["action"], action["action"], action["amount"]))
        return actions

    @staticmethod
    def get_strategy_action(strategy, possible_actions) -> Tuple[str, int]:
        """
        choose an action based on the strategy.\n
        return (act, amt).
        """
        strategy_act_types = list(strategy.keys())
        possible_act_types = [t[0] for t in possible_actions]

        valid_act_types = list(set(strategy_act_types) & set(possible_act_types))
        probabilities = [strategy[a] for a in valid_act_types]

        select_act_type = random.choices(valid_act_types, weights=probabilities)[0]

        action = next((act, amt) for act_type, act, amt in possible_actions if act_type == select_act_type)
        return action


class MCCFRTrainer:
    def __init__(self):
        self.info_set = InfoSet()
        self.sb_amount = 5
        self.ante = 0
        self.initial_stack = 1000

        # round specific
        self.round_table = Table()
        self.round_updating_player = ""

    def train(self, iterations: int = 1, save_interval = 100):
        for i in range(iterations):
            start_time = time.time()
            self.start_new_round()
            print(f"round {i+1}/{iterations} finished in {time.time() - start_time:.2f} seconds")

            if save_interval and (i + 1) % save_interval == 0:
                print(f"Saving progress at iteration {i+1}...")
                self.info_set.save(f"mccfr_r{i+1}.json")
                print("Progress saved.")

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
        strategy = self.info_set.get_strategy(info_key, max(0.01, 0.1 * (0.99 ** depth)))

        actions = ActionBuilder.get_possible_actions(valid_actions, round_state['pot']['main']['amount'])

        if current_player == self.round_updating_player:
            util = defaultdict(float)
            node_util = 0
            
            for act_type, act, amt in actions:
                next_state, next_msgs = RoundManager.apply_action(state, act, amt)
                
                util[act_type] = self.traverse(next_state, next_msgs, reach_prop * strategy[act_type], depth + 1) 
                node_util += strategy[act_type] * util[act_type]

            for act_type, _, _ in actions:
                regret = util[act_type] - node_util
                self.info_set.update_regret_sum(info_key, act_type, regret)
                self.info_set.update_strategy_sum(info_key, act_type, reach_prop * strategy[act_type])

            return node_util
        
        else:
            act, amt = ActionBuilder.get_strategy_action(strategy, actions)
            next_state, next_msgs = RoundManager.apply_action(state, act, amt)

            for act_type, _, _ in actions:
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

    filepath = "mccfr_v6.json"
    if os.path.exists(filepath):
        print("mccfr_data file found in the current directory.")
        trainer.info_set.load(filepath)

    trainer.train(1000)
    trainer.info_set.save(filepath)
