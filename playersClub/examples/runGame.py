from ChefsHatPlayersClub.Agents.Classic import DQL
from ChefsHatPlayersClub.Agents.Classic import PPO
from ChefsHatGym.Agents import Agent_Naive_Random
from ChefsHatGym.env import ChefsHatEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import gym

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 10

"""Player Parameters"""
agent1 = Agent_Naive_Random.AgentNaive_Random("Random1")
agent2 = Agent_Naive_Random.AgentNaive_Random("Random2")
agent3 = PPO.PPO(name="PPO2", continueTraining=True, type="vsEveryone", initialEpsilon=1, verbose=True)  # training agent
agent4 = DQL.DQL(name="DQL3", continueTraining=True, type="vsEveryone", initialEpsilon=1, verbose=True)  # training agent
agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
playersAgents = [agent1, agent2, agent3, agent4]

rewards = []
for r in playersAgents:
    rewards.append(r.getReward)

"""Experiment parameters"""
saveDirectory = "/home/pablo/Documents/Datasets/ChefsHatCompetition/playerClubTesting/"
verbose = False
saveLog = False
saveDataset = False
episodes = 1


"""Setup environment"""
env = gym.make('chefshat-v0') #starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Train Agent"""
for a in range(episodes):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction":False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

        currentPlayer.actionUpdate(observations, nextobs, action, reward, info)

        if isMatchOver:
            for p in playersAgents:
                p.matchUpdate(info)
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")

"""Evaluate Agent"""
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)
"""Start Environment"""
for a in range(episodes):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction":False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

        if isMatchOver:
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")