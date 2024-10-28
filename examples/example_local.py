from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP
from ChefsHatPlayersClub.agents.chefs_cup_v2.larger_value.larger_value import (
    AgentLargerValue,
)


# Room parameters
room_name = "1000_DQL_vs_Random_random_random_random"

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 10
maxRounds = -1

# Logging information
verbose_console = True
verbose_log = True
game_verbose_console = True
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

# Create the players
p1 = AgentDQL(
    name="01",
    verbose_log=agentVerbose,
    agentType="vsEveryone",
    continueTraining=False,
    log_directory=logDirectory,
)
p2 = AgentPPO(
    name="02",
    verbose_log=agentVerbose,
    agentType="vsEveryone",
    continueTraining=False,
    log_directory=logDirectory,
)

p3 = AgentLargerValue(
    name="03",
    verbose_log=agentVerbose,
    log_directory=logDirectory,
)

p4 = AIACIMP(
    name="03",
    verbose_log=agentVerbose,
    continueTraining=False,
    log_directory=logDirectory,
)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)

# Start the game
info = room.start_new_game()

print(f"Performance score: {info['Game_Performance_Score']}")
print(f"Scores: {info['Game_Score']}")
