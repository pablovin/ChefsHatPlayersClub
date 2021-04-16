from ChefsHatGym.Agents import IAgent
import numpy
import copy

from keras.layers import Input, Dense, Lambda, Multiply
import keras.backend as K

from keras.models import Model
from keras.optimizers import Adam

from keras.models import load_model

from ChefsHatPlayersClub.Agents.Util import MemoryBuffer

import random

from ChefsHatGym.Rewards import RewardOnlyWinning

import os
import sys

import urllib.request

types = ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]

class DQL(IAgent.IAgent):

    name="DQL_"
    actor = None
    training = False


    loadFrom = {"vsRandom":"Trained/dql_vsRandom.hd5",
            "vsEveryone":"Trained/dql_vsEveryone.hd5",
                "vsSelf":"Trained/dql_vsSelf.hd5",}
    downloadFrom = {"vsRandom":"https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsRandom.hd5",
            "vsEveryone":"https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsEveryone.hd5",
                "vsSelf":"https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsSelf.hd5",}

    def __init__(self, name, continueTraining=False, type="Scratch", initialEpsilon=1, loadNetwork="", saveFolder="", verbose=False):
        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.name += type+"_"+name
        self.loadNetwork = loadNetwork
        self.saveModelIn = saveFolder
        self.verbose = verbose

        self.type = type
        self.reward = RewardOnlyWinning.RewardOnlyWinning()

        self.startAgent()


        if not type=="Scratch":
            fileName = os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6]+self.loadFrom[type]
            if not os.path.exists(os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6]+"/Trained/"):
                os.mkdir(os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6]+"/Trained/")

            if not os.path.exists(fileName):
                urllib.request.urlretrieve(self.downloadFrom[type], fileName)
            # fileName = "/home/pablo/Documents/Workspace/ChefsHatPlayersClub/venv/lib/python3.6/site-packages/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsRandom.hd5"
            self.loadModel(fileName)

        if not loadNetwork =="":
            self.loadModel(loadNetwork)


    def getReward(self, info, stateBefore, stateAfter):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)


    def startAgent(self):

        self.hiddenLayers = 1
        self.hiddenUnits = 256
        self.batchSize = 10
        self.tau = 0.52  # target network update rate

        self.gamma = 0.95  # discount rate
        self.loss = "mse"

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990


        # self.tau = 0.1 #target network update rate

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0 #no exploration while testing

        # behavior parameters
        self.prioritized_experience_replay = False
        self.dueling = False

        QSize = 20000
        self.memory = MemoryBuffer.MemoryBuffer(QSize, self.prioritized_experience_replay)

        # self.learning_rate = 0.01
        self.learning_rate = 0.001

        self.buildModel()



    def buildModel(self):

          self.buildSimpleModel()

          self.actor.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

          self.targetNetwork.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])


    def buildSimpleModel(self):

        """ Build Deep Q-Network
               """

        def model():
            inputSize = 28
            inputLayer = Input(shape=(inputSize,),
                               name="State")  # 5 cards in the player's hand + maximum 4 cards in current board

            # dense = Dense(self.hiddenLayers, activation="relu", name="dense_0")(inputLayer)
            for i in range(self.hiddenLayers + 1):

                if i == 0:
                    previous = inputLayer
                else:
                  previous = dense

                dense = Dense(self.hiddenUnits * (i + 1), name="Dense" + str(i), activation="relu")(previous)

            if (self.dueling):
                # Have the network estimate the Advantage function as an intermediate layer
                dense = Dense(self.outputSize + 1, activation='linear', name="duelingNetwork")(dense)
                dense = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                           output_shape=(200,))(dense)


            possibleActions = Input(shape=(200,),
                       name="PossibleAction")


            dense = Dense(200, activation='softmax')(dense)
            output = Multiply()([possibleActions, dense])

            # probOutput =  Dense(self.outputSize, activation='softmax')(dense)

            return Model([inputLayer, possibleActions], output)

        self.actor = model()
        self.targetNetwork =  model()

    def getAction(self, observations):

        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(stateVector, 0)
        possibleActions = numpy.array(possibleActions)

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

    def loadModel(self, model):

        self.actor  = load_model(model)
        self.targetNetwork = load_model(model)


    def updateTargetNetwork(self):

        W = self.actor.get_weights()
        tgt_W = self.targetNetwork.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.targetNetwork.set_weights(tgt_W)



    def updateModel(self, game, thisPlayer):

        """ Train Q-network on batch sampled from the buffer
                """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, possibleActions, newPossibleActions, idx = self.memory.sample_batch(self.batchSize)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.actor.predict([s, possibleActions])
        next_q = self.actor.predict([new_s, newPossibleActions])
        q_targ = self.targetNetwork.predict([new_s, newPossibleActions])

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = numpy.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]


            if (self.prioritized_experience_replay):
                # Update PER Sum Tree
                self.memory.update(idx[i], abs(old_q - q[i, a[i]]))


        # Train on batch
        history = self.actor.fit([s,possibleActions] , q, verbose=False)

        if (game + 1) % 5 == 0 and not self.saveModelIn == "":
            self.actor.save(self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_"+str(thisPlayer)+".hd5")

        if self.verbose:
            print ("-- "+self.name + ": Epsilon:" + str(self.epsilon) + " - Loss:" + str(history.history['loss']))


    def memorize(self, state, action, reward, next_state, done, possibleActions, newPossibleActions):

        if (self.prioritized_experience_replay):
            state = numpy.expand_dims(numpy.array(state), 0)
            next_state = numpy.expand_dims(numpy.array(next_state), 0)
            q_val = self.actor.predict(state)
            q_val_t = self.targetNetwork.predict(next_state)
            next_best_action = numpy.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0


        self.memory.memorize(state, action, reward, done, next_state, possibleActions, newPossibleActions, td_error)



    def actionUpdate(self, observation, nextObservation, action, reward, info):

        if self.training:
            done = info["thisPlayerFinished"]

            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            next_state = numpy.concatenate((nextObservation[0:11], nextObservation[11:28]))
            newPossibleActions = nextObservation[28:]

            action = numpy.argmax(action)
            self.memorize(state, action, reward, next_state, done, possibleActions, newPossibleActions)


    def matchUpdate(self, info):

        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            if self.memory.size() > self.batchSize:
                self.updateModel(rounds, thisPlayer)
                self.updateTargetNetwork()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay



