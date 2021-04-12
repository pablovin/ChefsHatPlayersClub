from ChefsHatGym.Evaluation import Tournament
from ChefsHatGym.env import ChefsHatEnv
import random
from Agents import Agent_DQL
import time


"""Agents Factory"""
def getAgents(totalAgents):
    agents = []
    for i in range(totalAgents):
        agents.append(Agent_DQL.AgentDQL(name="DQL_"+str(i), training=True, initialEpsilon=1))  # training agent

    random.shuffle(agents)
    return agents


start_time = time.time()
"""Tournament parameters"""
saveTournamentDirectory = "/home/pablo/Documents/Datasets/ChefsHatCompetition/testing" #Where all the logs will be saved
agents = getAgents(2)

tournament = Tournament.Tournament(agents, saveTournamentDirectory, verbose=True, threadTimeOut=15, actionTimeOut=5, gameType=ChefsHatEnv.GAMETYPE["POINTS"], gameStopCriteria=15)
first, second, third, fourth = tournament.runTournament()

print("--- %s seconds ---" % (time.time() - start_time))
print ("Winner:" + str(first.name))
print ("Second:" + str(second.name))
print ("Third:" + str(third.name))
print ("Fourth:" + str(fourth.name))


