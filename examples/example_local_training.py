from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon

# from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.chefs_cup_v2.larger_value.larger_value import (
    AgentLargerValue,
)

# Room parameters
room_name = "Testing_2_Local"

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 3
maxRounds = -1

# Logging information
verbose_console = True
verbose_log = True
game_verbose_console = False
game_verbose_log = True
save_dataset = True


# Start the room
room = ChefsHatRoomLocal(
    room_name,
    game_type=game_type,
    stop_criteria=stop_criteria,
    max_rounds=maxRounds,
    verbose_console=verbose_console,
    verbose_log=verbose_log,
    game_verbose_console=game_verbose_console,
    game_verbose_log=game_verbose_log,
    save_dataset=save_dataset,
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
    verbose_log=True,
    log_directory=logDirectory,
)
p3 = AgentRandon(name="random_1", verbose_log=agentVerbose, log_directory=logDirectory)
p4 = AgentRandon(name="random_2", verbose_log=agentVerbose, log_directory=logDirectory)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)

# Start the game
info = room.start_new_game()

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
