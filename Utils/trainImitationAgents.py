from ChefsHatGym.Agents import Agent_Naive_Random
from ChefsHatGym.env import ChefsHatEnv

from ChefsHatPlayersClub.Agents.KarmaChameleonClub import AIRL

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]
gameStopCriteria = 100

demonstrationsTraining = "/home/pablo/Documents/Datasets/CHefsHatWeb/Online2/Neurips_Rivalry/demonstrations/training"
demonstrationTesting = "/home/pablo/Documents/Datasets/CHefsHatWeb/Online2/Neurips_Rivalry/demonstrations/testing"

saveModel = "/home/pablo/Documents/Datasets/CHefsHatWeb/Online2/Neurips_Rivalry/newHumanModels/"


for personFile in os.listdir(demonstrationsTraining):

    demonstration = demonstrationsTraining+"/"+personFile
    if not os.path.exists(saveModel+"/"+personFile):
        os.makedirs(saveModel+"/"+personFile)
        """Player Parameters"""
        agent1 = Agent_Naive_Random.AgentNaive_Random("Random1")
        agent2 = Agent_Naive_Random.AgentNaive_Random("Random2")
        agent3 = Agent_Naive_Random.AgentNaive_Random("Random3")
        agent4 = AIRL.AIRL(name=str(personFile), continueTraining=True, type="Scratch", demonstrations=demonstration, saveFolder =saveModel+"/"+personFile, initialEpsilon=1, verbose=True)
        agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
        playersAgents = [agent1, agent2, agent3, agent4]

        rewards = []
        for r in playersAgents:
            rewards.append(r.getReward)

        """Experiment parameters"""
        saveDirectory = "/home/pablo/Documents/Datasets/CHefsHatWeb/Online2/Neurips_Rivalry/savedHumanModels/Experiments"
        verbose = False
        saveLog = True
        saveDataset = False

        """Setup environment"""
        env = gym.make('chefshat-v0') #starting the game Environment
        env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria,
                            playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True,
                            saveLog=saveLog)

        """Start Environment"""
        observations = env.reset()
        while not env.gameFinished:
            currentPlayer = playersAgents[env.currentPlayer]

            observations = env.getObservation()
            action = currentPlayer.getAction(observations)

            info = {"validAction": False}
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