#Adapted from: https://github.com/germain-hug/Deep-RL-Keras


from ChefsHatGym.Agents import IAgent
import numpy
import copy

from keras.layers import Input, Dense,  Multiply
from keras.models import Model
from keras.optimizers import Adam

import keras.backend as K

from keras.models import load_model

import tensorflow.compat.v1 as tfc

import random

from ChefsHatGym.Rewards import RewardOnlyWinning

import os
import sys
import urllib

def actorLoss():

    def loss(y_true, y_pred):

        y_tru_valid = y_true[:, 0:200]
        advantage = y_true[:, 200][0]

        weighted_actions = K.sum(y_tru_valid * y_pred, axis=0)
        eligibility = -tfc.log(weighted_actions + 1e-5) * advantage
        entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=1)
        return K.mean(0.001 * entropy - K.sum(eligibility))

    return loss


class A2C(IAgent.IAgent):
    name = "PPO_"
    actor = None
    training = False


    loadFrom = {"vsRandom":["Trained/a2c_actor_vsRandom.hd5","Trained/a2c_critic_vsRandom.hd5"],
            "vsEveryone":["Trained/a2c_actor_vsEveryone.hd5","Trained/a2c_critic_vsEveryone.hd5"],
                "vsSelf":["Trained/a2c_actor_vsSelf.hd5","Trained/a2c_critic_vsSelf.hd5"]}
    downloadFrom = {"vsRandom":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_actor_vsRandom.hd5,"
                                "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_critic_vsRandom.hd5"],
            "vsEveryone":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_actor_vsEveryone.hd5",
                          "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_critic_vsEveryone.hd5"],
                "vsSelf":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_actor_vsSelf.hd5,"
                          "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/a2c_critic_vsSelf.hd5"],}



    def __init__(self,name, continueTraining=False, type="Scratch", initialEpsilon=1, loadNetwork="", saveFolder="", verbose=False):
        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.name += type + "_" + name
        self.loadNetwork = loadNetwork
        self.saveModelIn = saveFolder
        self.verbose = verbose

        self.type = type
        self.reward = RewardOnlyWinning.RewardOnlyWinning()

        self.startAgent()

        if not type == "Scratch":
            fileNameActor = os.path.abspath(sys.modules[A2C.__module__].__file__)[0:-6] + self.loadFrom[type][0]
            fileNameCritic = os.path.abspath(sys.modules[A2C.__module__].__file__)[0:-6] + self.loadFrom[type][1]
            if not os.path.exists(os.path.abspath(sys.modules[A2C.__module__].__file__)[0:-6] + "/Trained/"):
                os.mkdir(os.path.abspath(sys.modules[A2C.__module__].__file__)[0:-6] + "/Trained/")

            if not os.path.exists(fileNameCritic):
                urllib.request.urlretrieve(self.downloadFrom[type][0], fileNameActor)
                urllib.request.urlretrieve(self.downloadFrom[type][1], fileNameCritic)
            self.loadModel([fileNameActor,fileNameCritic])

        if not loadNetwork == "":
            self.loadModel(loadNetwork)



    def startAgent(self):

        #My Estimation
        self.hiddenLayers = 2
        self.hiddenUnits = 64
        self.gamma = 0.95  # discount rate

        self.outputActivation = "linear"
        self.loss = "mse"

        self.memory = []

        #Game memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.possibleActions = []


        self.learning_rate = 0.001

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0 #no exploration while testing

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990


        self.buildModel()


    def buildModel(self):

        inputSize = 28
        #shared part of the model
        inp = Input((inputSize, ), name="State")


        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(self.hiddenUnits * (i + 1), name="Dense" + str(i), activation="relu")(previous)

        #Actor network
        densea1 = Dense(128, activation='relu', name="actor_dense_3")(dense)

        outActor = Dense(200, activation='softmax', name="Actor_output")(densea1)

        possibleActions = Input(shape=(200,),
                                name="PossivleActions")
        outputPossibleActor = Multiply()([possibleActions, outActor])

        #Critic network
        densec1 = Dense(128, activation='relu', name="critic_dense_3")(dense)
        outCritic = Dense(1, activation='linear', name="critic_Output")(densec1)

        #build networks
        self.critic = Model(inp, outCritic)
        self.actor = Model([inp,possibleActions] , outputPossibleActor)

        self.getOptmizers()



    def getOptmizers(self):


        rmsOptmizer = Adam(lr=self.learning_rate)

        #optmizers

        #optmizer actor

        """ Actor Optimization: Advantages + Entropy term to encourage exploration
                (Cf. https://arxiv.org/abs/1602.01783)
                """

        action_pl= K.placeholder(shape=(None, self.outputSize))
        advantage_pl = K.placeholder(shape=(None,))
        k_constants = K.variable(0)

        weighted_actions = K.sum(action_pl * self.actor.output, axis=0)
        eligibility = -tfc.log(weighted_actions + 1e-5) * advantage_pl
        entropy = K.sum(self.actor.output * K.log(self.actor.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)

        updates = rmsOptmizer.get_updates(self.actor.trainable_weights, [], loss)

        self.actorOptmizer = K.function([self.actor.input, action_pl, advantage_pl], loss, updates=updates)


        #Critic optmizer

        """ Critic Optimization: Mean Squared Error over discounted rewards
                """

        discounted_r = K.placeholder(shape=(None,))
        critic_loss = K.mean(K.square(discounted_r - self.critic.output))
        updates = rmsOptmizer.get_updates(self.critic.trainable_weights, [], critic_loss)
        self.criticOptmizer = K.function([self.critic.input, discounted_r], critic_loss, updates=updates)


    def getReward(self, info):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)



    def getAction(self, observations):

        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(numpy.array(stateVector), 0)
        possibleActions2 = copy.copy(possibleActions)

        if numpy.random.rand() <= self.epsilon:

            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[0].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1

        else:
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)
            a = self.actor.predict([stateVector, possibleActionsVector])[0]

        return a



    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = numpy.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r


    def loadModel(self, model):
        actorModel, criticModel = model
        self.actor  = load_model(actorModel)
        self.critic = load_model(criticModel)
        self.getOptmizers()


    def updateModel(self,  game, thisPlayer):

        self.memory = numpy.array(self.memory)

        state =  numpy.array(self.states)
        action = self.actions
        reward = self.rewards
        possibleActions = numpy.array(self.possibleActions)


        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(reward)
        state_values = self.critic.predict(numpy.array(state))
        advantages = discounted_rewards - numpy.reshape(state_values, len(state_values))


        actorLoss = self.actorOptmizer([[state, possibleActions], action, advantages])

        actorLoss = numpy.average(actorLoss)

        criticLoss = self.criticOptmizer([state, discounted_rewards])

        criticLoss = numpy.mean(criticLoss)

        self.losses.append([actorLoss,criticLoss])

        # print ("Loss Critic:" + str(history.history['loss']))

        #Update the decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        #save model
        if (game + 1) % 100 == 0 and not(self.saveModelIn ==""):
            self.actor.save(self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_"+str(thisPlayer)+".hd5")
            self.critic.save(self.saveModelIn + "/critic_iteration_"  + str(game) + "_Player_"+str(thisPlayer)+".hd5")

        if self.verbose:
            print ("-- "+self.name + ": Epsilon:" + str(self.epsilon) + " - ALoss:" + str(actorLoss) + " - " + "CLoss: " + str(criticLoss))


    def resetMemory (self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.possibleActions = []

    def matchUpdate(self, info):

        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            self.updateModel(rounds, thisPlayer)
            self.resetMemory()

    def actionUpdate(self, observation, nextObservation, action, reward, info):

        state = numpy.concatenate((observation[0:11], observation[11:28]))
        possibleActions = observation[28:]

        if self.training:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.possibleActions.append(possibleActions)





