

from Agents import Agent_Naive_Random, Agent_Embedded


from Rewards import RewardOnlyWinning

from env import ChefsHatEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from stable_baselines3 import PPO


import gym

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["NUMGAMES"]
gameStopCriteria = 100

"""Player Parameters"""
agent1 = Agent_Naive_Random.AgentNaive_Random("Random1")
agent2 = Agent_Naive_Random.AgentNaive_Random("Random2")
agent3 = Agent_Naive_Random.AgentNaive_Random("Random3")
agent4 = Agent_Embedded.AgentEmbedded("StableBaseline_PPO")  # training agent
agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
playersAgents = [agent1, agent2, agent3, agent4]

rewards = []
for r in playersAgents:
    rewards.append(r.getReward)

"""Experiment parameters"""
saveDirectory = "/home/pablo/Documents/Datasets/ChefsHatGym/ChefsHatGYM"
verbose = False
saveLog = False
saveDataset = False
episodes = 1

"""Setup environment"""
env = gym.make('chefshat-v0') #starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start the agent and train it"""
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

"""Update the embedded model"""
agent4.model = model


"""Start a new experiment for testing"""
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames, logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

for a in range(episodes):

    obs = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]
        obs = env.getObservation()

        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction": False}
        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

        if isMatchOver:
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")


