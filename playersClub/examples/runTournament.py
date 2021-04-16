from ChefsHatGym.Evaluation import Tournament
from ChefsHatGym.env import ChefsHatEnv
import random
from ChefsHatPlayersClub.Agents.Classic import DQL
from ChefsHatPlayersClub.Agents.Classic import PPO
from ChefsHatPlayersClub.Agents.KarmaChameleonClub import AIRL


import time


"""Agents Factories"""
def getAllAgents(compType):
    agents = []

    for type in ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]:
        for learning in (True, False):
            agents.append(DQL.DQL(name=str(compType)+"_L_"+str(learning), continueTraining=learning, type=type, initialEpsilon=1, verbose=False) )
            agents.append(PPO.PPO(name=str(compType)+"_L_"+str(learning), continueTraining=learning, type=type, initialEpsilon=1, verbose=False) )


    for type in AIRL.types:
        print ("Type:" + str(type))
        agents.append(AIRL.AIRL(name=str(compType), continueTraining=False, type=type, initialEpsilon=1, verbose=True) )



    random.shuffle(agents)
    return agents


start_time = time.time()
"""Tournament parameters"""
saveTournamentDirectory = "/home/pablo/Documents/Datasets/ChefsHatCompetition/tournamentAllTesting/" #Where all the logs will be saved

"Competitive Agents"
print ("Loading Comp Agents...")
compAgents = getAllAgents("COOMP")
# print ("Loading COOP Agents...")
# coopAgents = getAllAgents("COOP")
# print ("Loading COMPCOOP Agents...")
# compCoopAgents = getAllAgents("COMPCOOP")


tournament = Tournament.Tournament(saveTournamentDirectory, oponentsCompCoop=compAgents, verbose=True, threadTimeOut=5, actionTimeOut=5, gameType=ChefsHatEnv.GAMETYPE["POINTS"], gameStopCriteria=15)
tournament.runTournament()

print("--- %s seconds ---" % (time.time() - start_time))


