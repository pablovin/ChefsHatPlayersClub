from ChefsHatGym.gameRooms.chefs_hat_room_remote import ChefsHatRoomRemote
from ChefsHatGym.env import ChefsHatEnv

from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.ainsa import AINSA

import redis
import time

# Create the players
p1 = AgentDQL(
    name="01",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
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

p3 = AINSA(
    name="03",
    continueTraining=False,
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=True,
)

p4 = AIACIMP(
    "04",
    continueTraining=False,
    demonstrations="",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory="",
)

# Clean all the rooms
r = redis.Redis()
r.flushall()

# Room parameters
room_name = "Testing_1_Remote"
timeout_player_subscribers = 200
timeout_player_response = 5
verbose = True

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 2
maxRounds = 5

# Start the room
room = ChefsHatRoomRemote(
    room_name,
    timeout_player_subscribers=timeout_player_subscribers,
    timeout_player_response=timeout_player_response,
    game_type=game_type,
    stop_criteria=stop_criteria,
    max_rounds=maxRounds,
    verbose=verbose,
)

# Give enought time for the room to setup
time.sleep(1)


# Join agents
for p in [p1, p2, p3, p4]:
    p.joinGame(room_name, verbose=False)

# Start the game
room.start_new_game(game_verbose=True)
while not room.get_room_finished():
    time.sleep(1)
