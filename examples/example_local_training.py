from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon

# from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.chefs_cup_v2.larger_value.larger_value import (
    AgentLargerValue,
)

# Room parameters
room_name = "Testing_1_Local"
timeout_player_response = 5
verbose = True

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 1000
maxRounds = -1

# Start the room
room = ChefsHatRoomLocal(
    room_name,
    timeout_player_response=timeout_player_response,
    game_type=game_type,
    stop_criteria=stop_criteria,
    max_rounds=maxRounds,
    verbose=verbose,
)

# Create the players
logDirectory = room.get_log_directory()
agentVerbose = True

p1 = AgentLargerValue(
    name="larger_value_1",
)

p2 = AgentPPO(
    "PPOvsRandom",
    continueTraining=True,
    agentType="Scratch",
    initialEpsilon=1,
    saveFolder=logDirectory,
    verbose=False,
    logDirectory=logDirectory,
)
p3 = AgentRandon(name="random_1", savelogDirectory=logDirectory, verbose=agentVerbose)
p4 = AgentRandon(name="random_2", savelogDirectory=logDirectory, verbose=agentVerbose)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)

# Start the game
info = room.start_new_game(game_verbose=True)

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
