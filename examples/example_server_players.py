from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL
from ChefsHatPlayersClub.agents.chefs_cup_v2.larger_value.larger_value import (
    AgentLargerValue,
)


room_pass = "password"
room_url = "localhost"
room_port = 10003


# Create the players
p1 = AgentDQL(
    name="01",
    verbose_console=True,
    verbose_log=True,
    agentType="vsEveryone",
    continueTraining=False,
)
p2 = AgentPPO(
    name="02",
    verbose_console=True,
    verbose_log=True,
    agentType="vsEveryone",
    continueTraining=False,
)
p3 = AgentAIRL(
    name="03",
    verbose_console=True,
    verbose_log=True,
    agentType="lil_abcd_",
    continueTraining=False,
)
p4 = AgentLargerValue(name="04", verbose_console=True, verbose_log=True)

# Join agents

p1.joinGame(room_pass=room_pass, room_url=room_url, room_port=room_port)
p2.joinGame(room_pass=room_pass, room_url=room_url, room_port=room_port)
p3.joinGame(room_pass=room_pass, room_url=room_url, room_port=room_port)
p4.joinGame(room_pass=room_pass, room_url=room_url, room_port=room_port)
