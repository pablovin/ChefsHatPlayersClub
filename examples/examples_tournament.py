from ChefsHatGym.gameRooms.chefs_hat_tournament import ChefsHatRoomTournament
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatPlayersClub.agents.classic.dql import AgentDQL
from ChefsHatPlayersClub.agents.classic.ppo import AgentPPO
from ChefsHatPlayersClub.agents.karma_camaleon_club.airl import AgentAIRL
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.aiacimp import AIACIMP
from ChefsHatPlayersClub.agents.chefs_cup_v1.team_yves.ainsa import AINSA


# Tounament parameters
tournamentFolder = "temp_tournament/"
tournament_name = "Tournament_1"
timeout_player_response = 5
verbose = True

# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 3
maxRounds = -1

# Create the players
p1 = AgentDQL(
    name="01",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)

p2 = AgentPPO(
    "02",
    continueTraining=False,
    agentType="vsRandom",
    initialEpsilon=1,
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

p5 = AgentAIRL(
    "05",
    continueTraining=False,
    agentType="lilAle",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)


p6 = AgentAIRL(
    "06",
    continueTraining=False,
    agentType="lilDJ",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)

p7 = AgentAIRL(
    "07",
    continueTraining=False,
    agentType="lil_abcd_",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)

p8 = AgentDQL(
    name="08",
    continueTraining=False,
    agentType="vsSelf",
    initialEpsilon=0.2,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)

p9 = AgentPPO(
    "09",
    continueTraining=False,
    agentType="vsSelf",
    initialEpsilon=1,
    loadNetwork="",
    saveFolder="",
    verbose=False,
    logDirectory=tournamentFolder,
)


# Start the tournament
tournament = ChefsHatRoomTournament(
        oponents = [p1,p2,p3,p4, p5, p6, p7, p8 ,p9],
        tournament_name = tournament_name,
        game_type = game_type,
        stop_criteria= stop_criteria,
        max_rounds = maxRounds,
        verbose= verbose,
        save_dataset = True,
        save_game_log = True,
        log_directory = tournamentFolder
)


# Run the tournament
tournament.runTournament()