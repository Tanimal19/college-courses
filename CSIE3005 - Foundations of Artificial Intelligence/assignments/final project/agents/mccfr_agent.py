from game.players import BasePokerPlayer
from train import InfoSet, ActionBuilder
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
import random
import os

class MCCFRAgent(
    BasePokerPlayer
):
    def __init__(self):
        super().__init__()
        self.info_set = InfoSet()
        self.round = 0

        filepath = "v7_500k.pkl"
        if os.path.exists(filepath):
            self.info_set.load(filepath)
        else:
            raise FileNotFoundError(f"{filepath} not found in the current directory!")
        
    def declare_action(self, valid_actions, hole_card, round_state):
        # if we have more than 1200 stack (1000 + 20 big blind), fold
        current_stack = round_state['seats'][round_state["next_player"]]['stack']
        if current_stack > 1200:
            return 'fold', 0
        
        info_key = self.info_set.build_info_key(hole_card, round_state, round_state['seats'][round_state["next_player"]]['uuid'])
        strategy = self.info_set.get_average_strategy(info_key)
        actions = ActionBuilder.get_possible_actions(valid_actions)
        act, amt = ActionBuilder.get_strategy_action(strategy, actions)

        # using monte carlo to regulate the action
        expected_value, pot_odds = self.__monte_carlo_agent(valid_actions, hole_card, round_state)
        if act == 'fold' and expected_value > 0.6 and expected_value > pot_odds:
            act = 'call'
            amt = valid_actions[1]['amount']
        elif act == 'call' and expected_value > 0.8 and expected_value > pot_odds:
            act = 'raise'
            amt = valid_actions[2]['amount']['max']
        elif (act == 'call' or act == 'raise') and expected_value < 0.3 and expected_value < pot_odds:
            return 'fold', 0

        # if opponent raise a lot and we have low win rate, we should fold
        if self.round < 15:
            street = round_state['street']
            last_action = round_state['action_histories'][street][-1]
            if last_action['action'] == 'RAISE' and last_action['amount'] > 800:
                if (act == 'call' or act == 'raise') and expected_value < 0.5:
                    return 'fold', 0

        return act, amt
    
    def __monte_carlo_agent(self, valid_actions, hole_card, round_state):
        pot = round_state['pot']['main']['amount']
        amount_to_call = valid_actions[1]['amount']
        pot_odds = amount_to_call / (pot + amount_to_call) if (pot + amount_to_call) > 0 else 1

        hole = [Card.from_str(c) for c in hole_card]
        community = [Card.from_str(c) for c in round_state['community_card']]

        # Simulate
        num_simulations = 500
        win, tie, loss = 0, 0, 0

        for i in range(num_simulations):
            deck = [Card.from_id(cid) for cid in range(1, 53)]
            appeared = hole + community
            unassigned = [card for card in deck if card not in appeared]
            
            opponent_hole = random.sample(unassigned, 2)
            unassigned = [card for card in unassigned if card not in opponent_hole]

            full_community = community + random.sample(unassigned, 5 - len(community)) 

            agent_strength = HandEvaluator.eval_hand(hole, full_community)
            opponent_strength = HandEvaluator.eval_hand(opponent_hole, full_community)

            if opponent_strength > agent_strength:
                loss += 1                
            elif opponent_strength < agent_strength:
                win += 1
            else:
                tie += 1

        win_prob = win / num_simulations
        tie_prob = tie / num_simulations
        expected_value = win_prob + 0.5 * tie_prob

        return expected_value, pot_odds

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round = round_count
        pass


    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MCCFRAgent()
