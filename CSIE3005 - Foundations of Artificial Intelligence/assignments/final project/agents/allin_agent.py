from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card


class AllinAgent(
    BasePokerPlayer
):
    
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

    def declare_action(self, valid_actions, hole_card, round_state):

        # if we have more than 1200 stack (1000 + 20 big blind), we fold
        if round_state['seats'][round_state["next_player"]]['stack'] > 1200:
            return 'fold', 0
            
        street = round_state['street']
        strength, rank = self.__calculate_hand_strength([Card.from_str(c) for c in hole_card], [Card.from_str(c) for c in round_state['community_card']])
            
        # if opponent raise a lot, and our hand is not strong enough, we fold
        last_action = round_state['action_histories'][street][-1]
        if last_action['action'] == 'RAISE' and last_action['amount'] > 800 and strength < 7:
            return 'fold', 0
        
        # we all in
        for action in valid_actions:
            if action["action"] == "raise":
                max_raise = action["amount"]["max"]
                return "raise", max_raise
        
        # if we cannot all in, we fold
        return "fold", 0
    
    def __calculate_hand_strength(self, hole_card, community_card):
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
    return AllinAgent()
