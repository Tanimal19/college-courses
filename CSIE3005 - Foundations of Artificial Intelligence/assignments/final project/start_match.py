import time
from collections import defaultdict
from game.game import setup_config, start_poker
from agents.monte_agent import setup_ai as monte_ai
from agents.mccfr_agent import setup_ai as mccfr_ai
from agents.allin_agent import setup_ai as allin_ai
from agents.random_player import setup_ai as random_ai
from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai
from baseline4 import setup_ai as baseline4_ai
from baseline5 import setup_ai as baseline5_ai
from baseline6 import setup_ai as baseline6_ai
from baseline7 import setup_ai as baseline7_ai

max_round = 10

player = (mccfr_ai, "agent")

opponents = [
    (baseline0_ai, "baseline0"),
    (baseline1_ai, "baseline1"),
    (baseline2_ai, "baseline2"),
    (baseline3_ai, "baseline3"),
    (baseline4_ai, "baseline4"),
    (baseline5_ai, "baseline5"),
    (baseline6_ai, "baseline6"),
    (baseline7_ai, "baseline7"),
    (random_ai, "random"),
    (allin_ai, "allin"),
]

win_count = defaultdict(int)

for opponent in opponents:
    print(f"\n=== {player[1]} vs. {opponent[1]} ===")
    win_count[opponent[1]] = 0

    for round in range(max_round):
        start_time = time.time()

        config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
        config.register_player(name="a", algorithm=player[0]())
        config.register_player(name="b", algorithm=opponent[0]())
        game_result = start_poker(config, verbose=0)

        players = game_result["players"]
        agent_stack = players[0]["stack"]
        opponent_stact = players[1]["stack"]
        if agent_stack > opponent_stact:
            win_count[opponent[1]] += 1

        print(f"game {round+1}, stack: {agent_stack}-{opponent_stact}, time: {time.time() - start_time:.2f}s")


print("\n=== Results ===")
for opponent_name, count in win_count.items():
    print(f"{opponent_name}: {count}/{max_round}")


# nohup python -u start_match.py &> match.log &