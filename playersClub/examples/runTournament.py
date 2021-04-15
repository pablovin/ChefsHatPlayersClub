from ChefsHatGym.Evaluation import Tournament
from ChefsHatGym.env import ChefsHatEnv
import random
from src.ChefsHatPlayersClub.Agents.Classic import Agent_DQL
import time


"""Agents Factory"""
def getAgents(totalAgents):
    agents = []
    for i in range(totalAgents):
        agents.append(Agent_DQL.AgentDQL(name="DQL_" + str(i), training=True, initialEpsilon=1))  # training agent

    random.shuffle(agents)
    return agents


start_time = time.time()
"""Tournament parameters"""
saveTournamentDirectory = "/home/pablo/Documents/Datasets/ChefsHatCompetition/testing" #Where all the logs will be saved
agents = getAgents(17)

tournament = Tournament.Tournament(agents, saveTournamentDirectory, verbose=True, threadTimeOut=15, actionTimeOut=5, gameType=ChefsHatEnv.GAMETYPE["POINTS"], gameStopCriteria=15)
tournament.runTournament()

print("--- %s seconds ---" % (time.time() - start_time))


