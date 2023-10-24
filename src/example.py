from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL

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
p1 = AgentPPO(
    name="01",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
    logDirectory=logDirectory,
)
p2 = AgentDQL(
    name="02",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
    logDirectory=logDirectory,
)

p3 = AgentAIRL(
    name="03",
    continueTraining=False,
    agentType="lil_abcd_",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
    logDirectory=logDirectory,
)


p4 = AgentRandon(name="04", savelogDirectory=logDirectory, verbose=agentVerbose)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)


# Start the game
info = room.start_new_game(game_verbose=True)

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
