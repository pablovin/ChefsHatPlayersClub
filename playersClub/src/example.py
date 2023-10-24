from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL

# Room parameters
room_name = "Testing_1_Local"
timeout_player_response = 5
verbose = False

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 1
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
p1 = AgentDQL(name="01", continueTraining=True, agentType="vsRandom", initialEpsilon=1, loadNetwork="", saveFolder="", verbose=True, logDirectory=logDirectory)
p2 = AgentRandon(name="02", savelogDirectory=logDirectory,verbose=agentVerbose)
p3 = AgentRandon(name="03", savelogDirectory=logDirectory,verbose=agentVerbose)
p4 = AgentRandon(name="04", savelogDirectory=logDirectory,verbose=agentVerbose)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)


# Start the game
info = room.start_new_game(game_verbose=False)

print(f"Performance score: {info['performanceScore']}")
print(f"Scores: {info['score']}")
