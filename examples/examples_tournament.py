from ChefsHatGym.gameRooms.chefs_hat_tournament import ChefsHatRoomTournament
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.ainsa import AINSA
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.allin import ALLIN
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.amyg4 import AMYG4
from ChefsHatPlayersClub.agents.chefs_cup_v2.bloom.Bloom import Bloom
from ChefsHatPlayersClub.agents.chefs_cup_v2.larger_value.larger_value import (
    AgentLargerValue,
)
from ChefsHatPlayersClub.agents.chefs_cup_v2.ppo_v2.ppo_v2 import AgentPPOV2

# Tounament parameters
tournamentFolder = "temp_tournament/"
tournament_name = "Tournament_1"
timeout_player_response = 5
verbose = True

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 3
maxRounds = -1

# Agents DQL and PPO

opponents = []
agentNumber = 0

opponents.append(Bloom("001", "", False, ""))
agentNumber += 1

opponents.append(AgentLargerValue("002"))
agentNumber += 1

opponents.append(AgentPPOV2("003"))
agentNumber += 1

for type in ["vsRandom", "vsEveryone", "vsSelf"]:
    opponents.append(
        AgentDQL(
            name=f"{agentNumber}",
            continueTraining=False,
            agentType=type,
            initialEpsilon=0.2,
            loadNetwork="",
            saveFolder="",
            verbose=False,
            logDirectory=tournamentFolder,
        )
    )

    opponents.append(
        AgentPPO(
            name=f"{agentNumber}",
            continueTraining=False,
            agentType=type,
            initialEpsilon=0.2,
            loadNetwork="",
            saveFolder="",
            verbose=False,
            logDirectory=tournamentFolder,
        )
    )

    agentNumber += 1

p1 = ALLIN(
    name="04",
    continueTraining=False,
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)

p2 = AMYG4(
    name="04",
    continueTraining=False,
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)
p3 = AIACIMP(
    "03",
    continueTraining=False,
    demonstrations="",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)


p4 = AINSA(
    name="04",
    continueTraining=False,
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)


for i in [p1, p2, p3, p4]:
    opponents.append(i)
    agentNumber += 1

agentNumbers = len(opponents)
for airl in [
    "lil_abcd_",
    "lilAbsol",
    "lilAle",
    "lilAna",
    "lilArkady",
    "lilAuar",
    "lilBlio1",
    "lilBlio2",
]:
    agentNumbers += 1
    opponents.append(
        AgentAIRL(
            f"0{agentNumbers}",
            continueTraining=False,
            agentType=airl,
            initialEpsilon=1,
            loadNetwork="",
            saveFolder="",
            verbose=False,
            logDirectory=tournamentFolder,
        )
    )


# Start the tournament
tournament = ChefsHatRoomTournament(
    oponents=opponents,
    tournament_name=tournament_name,
    game_type=game_type,
    stop_criteria=stop_criteria,
    max_rounds=maxRounds,
    verbose=verbose,
    save_dataset=True,
    save_game_log=True,
    log_directory=tournamentFolder,
)


# Run the tournament
tournament.runTournament()
