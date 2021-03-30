import numpy
from Agents import IAgent
import random
from Rewards import RewardOnlyWinning

class AgentEmbedded(IAgent.IAgent):

    def __init__(self, name="NAIVE_RANDOM", model=None):

        self.name = "EMBEDDED_"+name
        self.reward = RewardOnlyWinning.RewardOnlyWinning()
        self.model = model

    def getAction(self,  observations):

        action = self.model.predict(observations,deterministic=True)
        a = numpy.zeros(200)
        a[action[0]] = 1
        return a

    def train(self, observations, nextobs, action, reward, info):
        pass

    def getReward(self, info):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)
