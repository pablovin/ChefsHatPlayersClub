from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL

from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP
# from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.amyg4 import AMYG4

# Room parameters
room_name = "Testing_1_Local"
timeout_player_response = 5
verbose = True

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 2
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

p1 = AIACIMP(
    name="01",
    continueTraining=False,
    demonstrations="",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
    logDirectory=logDirectory,
)

p2 = AgentPPO(
    "02",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory="",
)
p3 = AgentDQL(
    name="03",
    continueTraining=False,
    agentType="vsEveryone",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
    logDirectory=logDirectory,
)

p4 = AgentAIRL(
    "04",
    continueTraining=False,
    agentType="lilAle",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory="",
)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)

# Start the game
info = room.start_new_game(game_verbose=True)

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
