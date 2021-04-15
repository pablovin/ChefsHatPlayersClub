from ChefsHatGym.Agents import IAgent
import numpy
import copy

from keras.layers import Input, Dense, Concatenate, Lambda, Multiply, LeakyReLU
import keras.backend as K

from keras.models import Model
from keras.optimizers import Adam

from keras.models import load_model

from Agents import MemoryBuffer

import random

class AIRL(IAgent.IAgent):

    name="AIRL_"

    type = {}
    def __init__(self, name, continueTraining, demonstrations=None, type="Scratch", initialEpsilon=1, loadNetwork="", saveFolder="", verbose=False):
        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.name += type + "_" + name
        self.loadNetwork = loadNetwork
        self.saveModelIn = saveFolder

        self.verbose = verbose
        self.demonstrations = numpy.load(demonstrations, allow_pickle=True)

        self.startAgent()



    def startAgent(self):
        numMaxCards, numCardsPerPlayer, actionNumber, loadModel, agentParams = 11, 28, 200, "", []

        self.numMaxCards = numMaxCards
        self.numCardsPerPlayer = numCardsPerPlayer
        self.outputSize = actionNumber  # all the possible ways to play cards plus the pass action

        self.hiddenLayers = 1
        self.hiddenUnits = 256
        self.batchSize = 128
        self.tau = 0.52  # target network update rate

        self.gamma = 0.95  # discount rate
        self.loss = "mse"

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95


        #Update the demonstrations side

        # print ("Shape:"+ str(self.demonstrations.shape))

        while self.demonstrations.shape[0] < self.batchSize:
            random.shuffle(self.demonstrations)
            # print("Shape:" + str(self.demonstrations.shape))
            # print("Shape:" + str(self.demonstrations[0].shape))
            self.demonstrations = numpy.append(self.demonstrations,numpy.expand_dims(self.demonstrations[0],0), axis=0)

        # print ("Shape:"+ str(self.demonstrations.shape))
        # input("here")
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

        if loadModel == "":
            self.buildModel()
        else:
            self.loadModel(loadModel)

    def getReward(self, info):

        state = numpy.concatenate((info["obsBefore"][0:11], info["obsBefore"][11:28]))

        rewardShape = numpy.concatenate([state,info["action"]])
        rewardShape = numpy.expand_dims(numpy.array(rewardShape), 0)
        reward = self.rewardNetwork.predict([rewardShape])[0][0]

        return reward


    def buildModel(self):

          self.buildSimpleModel()
          self.compileModels()

    def compileModels(self):
          self.actor.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

          self.targetNetwork.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

          self.rewardNetwork.compile(loss="binary_crossentropy", optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

          space = Input(shape=(28,))
          possibleAction = Input(shape=(200,))
          inputSelfReward = Input(shape=(228,))

          self.actor.trainable = False
          action = self.actor([space, possibleAction])

          concat = Concatenate()([space,action])

          self.combined = Model([space,possibleAction], self.rewardNetwork(concat))
          self.combinedDemonstrator = Model([inputSelfReward], self.rewardNetwork(inputSelfReward))

          self.combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate))
          self.combinedDemonstrator.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate))

    def buildSimpleModel(self):

        """ Build Deep Q-Network
               """

        def modelQValue():
            inputSize = self.numCardsPerPlayer + self.numMaxCards
            inputLayer = Input(shape=(28,),
                               name="State")  # 5 cards in the player's hand + maximum 4 cards in current board

            # dense = Dense(self.hiddenLayers, activation="relu", name="dense_0")(inputLayer)
            for i in range(self.hiddenLayers + 1):

                if i == 0:
                    previous = inputLayer
                else:
                  previous = dense

                dense = Dense(self.hiddenUnits * (i + 1), name="Dense" + str(i))(previous)
                dense = LeakyReLU()(dense)

            if (self.dueling):
                # Have the network estimate the Advantage function as an intermediate layer
                dense = Dense(self.outputSize + 1,  name="duelingNetwork")(dense)
                dense = LeakyReLU()(dense)
                dense = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                           output_shape=(self.outputSize,))(dense)

            possibleActions = Input(shape=(self.outputSize,),
                       name="PossibleAction")

            dense = Dense(self.outputSize, activation='softmax')(dense)
            output = Multiply()([possibleActions, dense])

            # probOutput =  Dense(self.outputSize, activation='softmax')(dense)

            return Model([inputLayer, possibleActions], output)

        def modelReward():
            inputSize = self.numCardsPerPlayer + self.numMaxCards
            inputLayer = Input(shape=(228,),
                               name="RewardInput")  # 5 cards in the player's hand + maximum 4 cards in current board

            dense = Dense(256, name="Dense" + str(0))(inputLayer)
            dense = LeakyReLU()(dense)
            # dense = Dense(256, name="Dense" + str(1), activation="relu")(dense)


            dense = Dense(1, activation='tanh')(dense)

            # probOutput =  Dense(self.outputSize, activation='softmax')(dense)

            return Model([inputLayer], dense)

        self.actor = modelQValue()
        self.targetNetwork =  modelQValue()
        self.rewardNetwork = modelReward()


    def getOptmizer(self):

        adamOptmizer = Adam(lr=self.learning_rate)

        state = K.placeholder(shape=(None, 28))
        nextState = K.placeholder(shape=(None, 28))
        actionProb =  K.placeholder(shape=(None, 200))

        state_d = K.placeholder(shape=(None, 28))
        nextState_d = K.placeholder(shape=(None, 28))
        actionProb_d =  K.placeholder(shape=(None, 200))

        gamma =  K.variable(self.gamma)

        stateValues = K.function([self.actor.input], self.actor.output)
        rewardValue = K.function([self.rewardNetwork.input], self.rewardNetwork.output)

        reward = rewardValue(state)
        stateValue = stateValues(state)
        nextStateValue = stateValue(nextState)

        reward_d = rewardValue(state_d)
        stateValue_d = stateValues(state_d)
        nextStateValue_d = stateValue(nextState_d)

        logits = reward + gamma * nextStateValue - stateValue - actionProb
        logits_d = reward_d+gamma*nextStateValue_d - stateValue_d - actionProb_d

        loss = K.mean(K.softplus(-(logits))) + K.mean(K.softplus((logits_d)))

        updatesOnline = adamOptmizer.get_updates(self.actor.trainable_weights, [], loss)
        updatesReward = adamOptmizer.get_updates(self.rewardNetwork.trainable_weights, [], loss)

        self.updateOnline = K.function([state, nextState, actionProb, state_d,nextState_d,actionProb_d], loss, updates=updatesOnline)
        self.updateReward = K.function([state, nextState, actionProb, state_d, nextState_d, actionProb_d], loss,
                                       updates=updatesReward)

    def getAction(self, observations):


        state = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActionsOriginal = observations[28:]

        stateVector = numpy.expand_dims(numpy.array(state), 0)

        possibleActions2 = copy.copy(possibleActionsOriginal)

        if numpy.random.rand() <= self.epsilon:
            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[0].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1

        else:
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)
            a = self.targetNetwork.predict([stateVector, possibleActionsVector])[0]

        return a

    def loadModel(self, model):

        onlineModel,rewardModel = model
        self.rewardNetwork = load_model(rewardModel)
        self.actor  = load_model(onlineModel)
        self.targetNetwork = load_model(onlineModel)
        self.compileModels()


    def updateTargetNetwork(self):

        W = self.actor.get_weights()
        tgt_W = self.targetNetwork.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.targetNetwork.set_weights(tgt_W)

        #
        # self.actor.set_weights(self.actor.get_weights())


    def updateModel(self, game, thisPlayer):

        """ Train Q-network on batch sampled from the buffer
                """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, possibleActions, newPossibleActions, idx = self.memory.sample_batch(self.batchSize)


        batchIndex = numpy.array(range(len(self.demonstrations)))
        random.shuffle(batchIndex)
        batchIndex = batchIndex[0:self.batchSize]

        # print ("Bacth Index:" + str(batchIndex))
        # print("Shape Demonstration Index:" + str(self.demonstrations.shape))
        # print("Shape Board Demonstrations:" + str(self.demonstrations[batchIndex, 0].shape))
        # print(" Board Demonstrations:" + str(self.demonstrations[batchIndex, 0]))
        # input("here")
        # board, playerHand, action
        # board = self.demonstrations[batchIndex,0]
        # # print ("Board:" + str(board))
        # # input("here")
        # playerHand = self.demonstrations[batchIndex,1]
        #
        # print ("Shape board:" + str(board.shape))
        # print ("Shape playerHand:" + str(playerHand.shape))
        d_s = self.demonstrations[batchIndex,0]
        # print ("Shape d_s:" + str(d_s.shape))
        d_a = self.demonstrations[batchIndex,1]
        # print ("D_a:"+str(d_a))
        # print ("Shape:" + str(d_a.shape))
        # input("here")
        # d_a = [numpy.argmax(i) for i in d_a]


        """
        Policy network generate trajectories
        """

        self.actor.trainable = False

        #Train on real data
        lossReward1 = self.combined.train_on_batch([s,possibleActions], numpy.ones(self.batchSize))

        #Train on demonstrator data
        d_action = numpy.zeros((self.batchSize,200))
        for x in range(self.batchSize):
            d_action[x][d_a[x]] = 1

        d_state = []
        for x in d_s:
            d_state.append(x)

        # print ("D_state:" + str(numpy.array(d_state).shape))
        # print("d_action:" + str(numpy.array(d_action).shape))
        featureInput = numpy.concatenate([d_state,d_action], axis=1)

        lossReward2 = self.combinedDemonstrator.train_on_batch([featureInput], numpy.zeros(self.batchSize))

        lossReward = 0.5*(lossReward1+lossReward2)

        #
        #
        # """
        # Obtain policy network outputs of current batch
        # """
        #
        # Apply Bellman Equation on batch samples to train our DDQN

        action = numpy.zeros((self.batchSize,200))
        for x in range(self.batchSize):
            action[x][a[x]] = 1

        # print ("s:" + str(numpy.array(s).shape))
        # print("action:" + str(numpy.array(action).shape))
        featureInput = numpy.concatenate([s,action], axis=1)

        new_r = self.rewardNetwork.predict([featureInput])
        q = self.actor.predict([s, possibleActions])
        next_q = self.actor.predict([new_s, newPossibleActions])
        q_targ = self.targetNetwork.predict([new_s, newPossibleActions])

        # self.successNetwork.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

        for i in range(s.shape[0]):
            if d[i]:
                q[i, a[i]] = new_r[i]
            else:
                next_best_action = numpy.argmax(next_q[i, :])
                q[i, a[i]] = new_r[i] + self.gamma * q_targ[i, next_best_action]


        # # Train on policy batch with new reward
        self.actor.trainable = True
        lossPolicy = self.actor.train_on_batch([s, possibleActions], q)[0]
        # lossReward2 = self.rewardNetwork.train_on_batch([d_s], numpy.ones(self.batchSize))
        #
        # lossPolicy = 0.5*(lossPolicy1+lossPolicy2)
        # lossReward = 0.5*(lossReward1+lossReward2)
        # self.losses.append([lossPolicy,lossReward])

        if (game + 1) % 5 == 0 and not self.saveModelIn=="":
            self.actor.save(self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_"+str(thisPlayer)+".hd5")
            self.rewardNetwork.save(
                self.saveModelIn + "/reward_iteration_" + str(game) + "_Player_" + str(thisPlayer) + ".hd5")
            self.lastModel = [self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_"+str(thisPlayer)+".hd5", self.saveModelIn + "/reward_iteration_" + str(game) + "_Player_" + str(thisPlayer) + ".hd5"]

        #
        print (" -- E:" + str(self.epsilon) + " - Lp:" + str(lossPolicy) + " - Lr" + str(lossReward))


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


    def train(self, observation, nextObservation, action, reward, info):


        rounds = info["rounds"]
        thisPlayer = info["thisPlayer"]
        done = info["thisPlayerFinished"]

        state = numpy.concatenate((observation[0:11], observation[11:28]))
        possibleActions = observation[28:]

        next_state = numpy.concatenate((nextObservation[0:11], nextObservation[11:28]))
        newPossibleActions = nextObservation[28:]


        if self.training:


            #memorize
            action = numpy.argmax(action)
            # state = numpy.expand_dims(numpy.array(state), 0)
            # next_state = numpy.expand_dims(numpy.array(next_state), 0)
            # possibleActions = numpy.expand_dims(numpy.array(possibleActions), 0)

            self.memorize(state, action, reward, next_state, done, possibleActions, newPossibleActions)

            #
            # self.memory.append((state, action, reward, next_state, done, possibleActions))

            # if self.memory.size() > self.batchSize:
            if self.memory.size() > self.batchSize and done:
                self.updateModel(rounds, thisPlayer)
                self.updateTargetNetwork()

                # Update the decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay





