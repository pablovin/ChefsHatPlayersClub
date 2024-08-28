# Adapted from: https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.rewards.reward import Reward
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning

from keras.layers import Input, Dense, Multiply
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

import random
import numpy
import copy
import os
import sys
import urllib.request
import tarfile
import tensorflow as tf

tf.experimental.numpy.experimental_enable_numpy_behavior()


class RewardLegal(Reward):
    rewardName = "RewardLegal"

    def getReward(self, thisPlayerPosition, matchFinished):
        pos_map = [1, 0.5, 0.25, -0.001]
        reward = -0.001
        if matchFinished:
            reward = pos_map[thisPlayerPosition]

        return reward


def proximal_policy_optimization_loss():
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        ENTROPY_LOSS = 5e-3
        y_tru_valid = y_true[:, 0:200]
        old_prediction = y_true[:, 200:400]
        advantage = y_true[:, 400][0]

        prob = K.sum(y_tru_valid * y_pred, axis=-1)
        old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(
                r * advantage,
                K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
                * advantage,
            )
            + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10))
        )

    return loss


# Adapted from: https://github.com/germain-hug/Deep-RL-Keras

types = ["Scratch", "vsRandom", "vsEveryone", "vsSelf"]


class AgentPPOV2(ChefsHatPlayer):
    suffix = "PPO_V2"
    actor = None
    training = False

    loadFrom = {
        "chefsHatV2": [
            "actor",
            "critic",
        ],
    }

    downloadFrom = {
        "chefsHatV2": [
            "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/src/ChefsHatPlayersClub/agents/chefs_cup_v2/ppo_v2/actor.tar",
            "https://github.com/pablovin/ChefsHatPlayersClub/blob/main/src/ChefsHatPlayersClub/agents/chefs_cup_v2/ppo_v2/critic.tar",
        ],
    }

    def __init__(
        self,
        name,
        continueTraining: bool = False,
        initialEpsilon: float = 1,
        loadNetwork="",
        saveFolder="",
        verbose_console: bool = False,
        verbose_log: bool = False,
        log_directory: str = "",
    ):
        super().__init__(
            self.suffix,
            name,
            this_agent_folder=saveFolder,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
        )

        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.loadNetwork = loadNetwork

        agentType = "chefsHatV2"
        self.type = agentType
        self.reward = RewardLegal()

        self.startAgent()

        fileNameActor = os.path.join(
            os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-6],
            "actor",
        )
        fileNameCritic = os.path.join(
            os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-6],
            "critic",
        )

        if not os.path.exists(fileNameCritic):
            urllib.request.urlretrieve(
                self.downloadFrom[agentType][0],
                os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-6],
            )
            urllib.request.urlretrieve(
                self.downloadFrom[agentType][1],
                os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-6],
            )

            with tarfile.open(fileNameActor + ".tar") as f:
                f.extractall(
                    os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-7],
                )

            with tarfile.open(fileNameCritic + ".tar") as f:
                f.extractall(
                    os.path.abspath(sys.modules[AgentPPOV2.__module__].__file__)[0:-7],
                )

        self.loadModel([fileNameActor, fileNameCritic])

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    # PPO Functions
    def startAgent(self):
        self.hiddenLayers = 2
        self.hiddenUnits = 64
        self.gamma = 0.95  # discount rate

        # Game memory
        self.resetMemory()

        self.learning_rate = 1e-4

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0  # no exploration while testing

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990

        self.buildModel()

    def buildActorNetwork(self):
        inputSize = 28
        inp = Input((inputSize,), name="Actor_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(
                self.hiddenUnits * (i + 1),
                name="Actor_Dense" + str(i),
                activation="relu",
            )(previous)

        outputActor = Dense(200, activation="softmax", name="actor_output")(dense)

        actionsOutput = Input(shape=(200,), name="PossibleActions")

        outputPossibleActor = Multiply()([actionsOutput, outputActor])

        self.actor = Model([inp, actionsOutput], outputPossibleActor)

        self.actor.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=[proximal_policy_optimization_loss()],
        )

    def buildCriticNetwork(self):
        # Critic model
        inputSize = 28

        inp = Input((inputSize,), name="Critic_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(
                self.hiddenUnits * (i + 1),
                name="Critic_Dense" + str(i),
                activation="relu",
            )(previous)

        outputCritic = Dense(1, activation="linear", name="critic_output")(dense)

        self.critic = Model([inp], outputCritic)

        self.critic.compile(Adam(self.learning_rate), "mse")

    def buildModel(self):
        self.buildCriticNetwork()
        self.buildActorNetwork()

    def discount(self, r):
        """Compute the gamma-discounted rewards over an episode"""
        discounted_r, cumul_r = numpy.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def loadModel(self, model):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
            ENTROPY_LOSS = 5e-3
            y_tru_valid = y_true[:, 0:200]
            old_prediction = y_true[:, 200:400]
            advantage = y_true[:, 400][0]

            prob = K.sum(y_tru_valid * y_pred, axis=-1)
            old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
            r = prob / (old_prob + 1e-10)

            return -K.mean(
                K.minimum(
                    r * advantage,
                    K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING)
                    * advantage,
                )
                + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10))
            )

        actorModel, criticModel = model
        self.actor = load_model(actorModel, custom_objects={"loss": loss})
        self.critic = load_model(criticModel, custom_objects={"loss": loss})

    def updateModel(self, game, thisPlayer):
        state = numpy.array(self.states)

        action = self.actions
        reward = numpy.array(self.rewards)
        possibleActions = numpy.array(self.possibleActions)
        realEncoding = numpy.array(self.realEncoding)

        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(reward)
        state_values = self.critic(numpy.array(state))
        advantages = discounted_rewards - numpy.reshape(state_values, len(state_values))

        criticLoss = self.critic.train_on_batch([state], [reward])

        actions = []
        for i in range(len(action)):
            advantage = numpy.zeros(numpy.array(action[i]).shape)
            advantage[0] = advantages[i]
            # print ("advantages:" + str(numpy.array(advantage).shape))
            # print ("actions:" + str(numpy.array(action[i]).shape))
            # print("realEncoding:" + str(numpy.array(realEncoding[i]).shape))
            concatenated = numpy.concatenate((action[i], realEncoding[i], advantage))
            actions.append(concatenated)
        actions = numpy.array(actions)

        actorLoss = self.actor.train_on_batch([state, possibleActions], [actions])

        self.loss_file.write(f"{criticLoss},{actorLoss}\n")

        # Update the decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if not self.saveModelIn == "":
            self.actor.save(
                os.path.join(
                    self.saveModelIn,
                    "actor_iteration_"
                    + str(game)
                    + "_Player_"
                    + str(thisPlayer)
                    + ".hd5",
                )
            )
            self.critic.save(
                os.path.join(
                    self.saveModelIn,
                    "critic_iteration_"
                    + str(game)
                    + "_Player_"
                    + str(thisPlayer)
                    + ".hd5",
                )
            )

        self.log(
            "-- "
            + self.name
            + ": Epsilon:"
            + str(self.epsilon)
            + " - ALoss:"
            + str(actorLoss)
            + " - "
            + "CLoss: "
            + str(criticLoss)
        )

    def resetMemory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.possibleActions = []
        self.realEncoding = []

    # Agent Chefs Hat Functions

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def get_reward(self, info):
        this_player = info["Author_Index"]
        this_player_name = info["Player_Names"][this_player]
        this_player_position = 3 - info["Match_Score"][this_player_name]
        this_player_finished = info["Finished_Players"][this_player_name]

        return self.reward.getReward(this_player_position, this_player_finished)

    def get_action(self, observations):
        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(numpy.array(stateVector), 0)
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

        return numpy.array(a)

    def update_end_match(self, info):
        if self.training:
            rounds = info["Rounds"]
            thisPlayer = info["Author_Index"]
            self.updateModel(rounds, thisPlayer)
            self.resetMemory()

    def update_my_action(self, info):
        if self.training:
            action_index = info["Action_Index"]
            observation = numpy.array(info["Observation_Before"])

            reward = self.get_reward(info)

            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            action = numpy.zeros(action.shape)
            action[action_index] = 1

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.possibleActions.append(possibleActions)
            self.realEncoding.append(action)
