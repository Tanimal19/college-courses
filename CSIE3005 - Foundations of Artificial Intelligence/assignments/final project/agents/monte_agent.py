from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
import random


class MonteAgent(
    BasePokerPlayer
):

    def declare_action(self, valid_actions, hole_card, round_state):

        # if we have more than 1200 stack (1000 + 20 big blind), fold
        current_stack = round_state['seats'][round_state["next_player"]]['stack']
        if current_stack > 1200:
            return 'fold', 0
        
        street = round_state['street']
        if street == 'preflop':
            # Preflop strategy: if we have a pair or high rank, call; otherwise fold
            hole = [Card.from_str(c) for c in hole_card]
            if hole[0].rank == hole[1].rank or (hole[0].rank >= 9 and hole[1].rank >= 9):
                return 'call', valid_actions[1]["amount"]

        
        pot = round_state['pot']['main']['amount']
        amount_to_call = valid_actions[1]['amount']
        pot_odds = amount_to_call / (pot + amount_to_call) if (pot + amount_to_call) > 0 else 1

        hole = [Card.from_str(c) for c in hole_card]
        community = [Card.from_str(c) for c in round_state['community_card']]

        # Simulate
        num_simulations = 1000
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

        # Strategy decision
        if expected_value > pot_odds and valid_actions[2]:
            if expected_value > 0.5:
                return 'raise', valid_actions[2]["amount"]["max"]
            else:
                return 'call', valid_actions[1]["amount"]
        else:                
            return 'fold', 0


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass


    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MonteAgent()