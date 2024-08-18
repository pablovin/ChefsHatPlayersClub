from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP

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

p1 = AIACIMP(
    name="01",
    continueTraining=False,
    verbose_log=True,
    logDirectory=logDirectory,
)

p2 = AgentPPO(
    "02",
    continueTraining=False,
    agentType="vsRandom",
    verbose_log=True,
    logDirectory=logDirectory,
)
p3 = AgentDQL(
    name="03",
    continueTraining=False,
    agentType="vsEveryone",
    verbose_log=True,
    logDirectory=logDirectory,
)

p4 = AgentAIRL(
    "04",
    continueTraining=False,
    agentType="lilAle",
    verbose_log=True,
    logDirectory=logDirectory,
)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)

# Start the game
info = room.start_new_game()

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
