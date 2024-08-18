from ChefsHatPlayersClub.agents.util.memory_buffer import MemoryBuffer

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning

import keras
from keras.layers import Input, Dense, Lambda, Multiply
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

import os
import sys
import urllib.request
import random
import numpy
import copy
from typing import Literal


types = ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]


class AgentDQL(ChefsHatPlayer):

    # _TYPES: Literal["Scratch", "vsRandom", "vsEveryone", "vsSelf"]

    suffix = "DQL"
    actor = None
    training = False

    loadFrom = {
        "vsRandom": "Trained/dql_vsRandom.hd5",
        "vsEveryone": "Trained/dql_vsEveryone.hd5",
        "vsSelf": "Trained/dql_vsSelf.hd5",
    }

    downloadFrom = {
        "vsRandom": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsRandom.hd5",
        "vsEveryone": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsEveryone.hd5",
        "vsSelf": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/classic/Trained/dql_vsSelf.hd5",
    }

    # self, a: int, b: str, c: float, type_: Literal["solar", "view", "both"] = "solar"):

    def __init__(
        self,
        name: str,
        continueTraining: bool = False,
        agentType: Literal["Scratch", "vsRandom", "vsEveryone", "vsSelf"] = "Scratch",
        initialEpsilon: int = 1,
        loadNetwork: str = "",
        saveFolder: str = "",
        verbose_console: bool = False,
        verbose_log: bool = False,
        log_directory: str = "",
    ):
        super().__init__(
            self.suffix,
            agentType + "_" + name,
            this_agent_folder=saveFolder,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
        )

        self.training = continueTraining
        self.initialEpsilon = initialEpsilon

        self.loadNetwork = loadNetwork

        self.type = agentType
        self.reward = RewardOnlyWinning()

        self.startAgent()

        if not agentType == "Scratch":
            fileName = os.path.join(
                os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[0:-6],
                self.loadFrom[agentType],
            )

            if not os.path.exists(
                os.path.join(
                    os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[0:-6],
                    "Trained",
                )
            ):
                os.mkdir(
                    os.path.join(
                        os.path.abspath(sys.modules[AgentDQL.__module__].__file__)[
                            0:-6
                        ],
                        "Trained",
                    )
                )

            if not os.path.exists(fileName):
                urlToDownload = self.downloadFrom[agentType]
                saveIn = fileName
                print(f"URL Download: {urlToDownload}")
                urllib.request.urlretrieve(self.downloadFrom[agentType], fileName)
            self.loadModel(fileName)

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    # DQL Functions
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
            self.epsilon = 0.0  # no exploration while testing

        # behavior parameters
        self.prioritized_experience_replay = False
        self.dueling = False

        QSize = 20000
        self.memory = MemoryBuffer(QSize, self.prioritized_experience_replay)

        # self.learning_rate = 0.01
        self.learning_rate = 0.001

        self.buildModel()

    def buildModel(self):
        self.buildSimpleModel()

        self.actor.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mse"],
        )

        self.targetNetwork.compile(
            loss=self.loss,
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mse"],
        )

    def buildSimpleModel(self):
        """Build Deep Q-Network"""

        def model():
            inputSize = 28
            inputLayer = Input(
                shape=(inputSize,), name="State"
            )  # 5 cards in the player's hand + maximum 4 cards in current board

            # dense = Dense(self.hiddenLayers, activation="relu", name="dense_0")(inputLayer)
            for i in range(self.hiddenLayers + 1):
                if i == 0:
                    previous = inputLayer
                else:
                    previous = dense

                dense = Dense(
                    self.hiddenUnits * (i + 1), name="Dense" + str(i), activation="relu"
                )(previous)

            if self.dueling:
                # Have the network estimate the Advantage function as an intermediate layer
                dense = Dense(
                    self.outputSize + 1, activation="linear", name="duelingNetwork"
                )(dense)
                dense = Lambda(
                    lambda i: K.expand_dims(i[:, 0], -1)
                    + i[:, 1:]
                    - K.mean(i[:, 1:], keepdims=True),
                    output_shape=(200,),
                )(dense)

            possibleActions = Input(shape=(200,), name="PossibleAction")

            dense = Dense(200, activation="softmax")(dense)
            output = Multiply()([possibleActions, dense])

            # probOutput =  Dense(self.outputSize, activation='softmax')(dense)

            return Model([inputLayer, possibleActions], output)

        self.actor = model()
        self.targetNetwork = model()

    def loadModel(self, model):
        self.actor = load_model(model)
        self.targetNetwork = load_model(model)

    def updateTargetNetwork(self):
        W = self.actor.get_weights()
        tgt_W = self.targetNetwork.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.targetNetwork.set_weights(tgt_W)

    def updateModel(self, game, thisPlayer):
        """Train Q-network on batch sampled from the buffer"""
        # Sample experience from memory buffer (optionally with PER)
        (
            s,
            a,
            r,
            d,
            new_s,
            possibleActions,
            newPossibleActions,
            idx,
        ) = self.memory.sample_batch(self.batchSize)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.actor([s, possibleActions])
        next_q = self.actor([new_s, newPossibleActions])
        q_targ = self.targetNetwork([new_s, newPossibleActions])

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = numpy.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]

            if self.prioritized_experience_replay:
                # Update PER Sum Tree
                self.memory.update(idx[i], abs(old_q - q[i, a[i]]))

        # Train on batch
        history = self.actor.fit([s, possibleActions], q, verbose=False)

        self.log(
            "-- "
            + self.name
            + ": Epsilon:"
            + str(self.epsilon)
            + " - Loss:"
            + str(history.history["loss"])
        )

    def memorize(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        possibleActions,
        newPossibleActions,
    ):
        if self.prioritized_experience_replay:
            state = numpy.expand_dims(numpy.array(state), 0)
            next_state = numpy.expand_dims(numpy.array(next_state), 0)
            q_val = self.actor(state)
            q_val_t = self.targetNetwork(next_state)
            next_best_action = numpy.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0

        self.memory.memorize(
            state,
            action,
            reward,
            done,
            next_state,
            possibleActions,
            newPossibleActions,
            td_error,
        )

    # Agent Chefs Hat Functions

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def get_action(self, observations):
        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(stateVector, 0)
        possibleActions = numpy.array(possibleActions)

        possibleActions2 = copy.copy(possibleActions)

        if numpy.random.rand() <= self.epsilon:
            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[
                0
            ].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1

        else:
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)

            a = self.actor([stateVector, possibleActionsVector])[0]

        return a

    def get_reward(self, info):
        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)

    def update_my_action(self, info):
        if self.training:
            action = numpy.array(info["action"])
            observation = numpy.array(info["observation"])
            nextObservation = numpy.array(info["nextObservation"])

            reward = self.get_reward(info)

            done = info["thisPlayerFinished"]

            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            next_state = numpy.concatenate(
                (nextObservation[0:11], nextObservation[11:28])
            )
            newPossibleActions = nextObservation[28:]

            action = numpy.argmax(action)
            self.memorize(
                state,
                action,
                reward,
                next_state,
                done,
                possibleActions,
                newPossibleActions,
            )

    def update_end_match(self, info):
        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            if self.memory.size() > self.batchSize:
                self.updateModel(rounds, thisPlayer)
                self.updateTargetNetwork()

                if not self.saveModelIn == "":

                    keras.models.save_model(
                        self.critic,
                        os.path.join(
                            self.saveModelIn,
                            "actor_Player" + str(self.name) + ".keras",
                        ),
                    )
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
