from ChefsHatGym.agents.chefs_hat_agent import ChefsHatAgent
import numpy
import random
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning


class AgentLargerValue(ChefsHatAgent):
    suffix = "LARGER_VALUE"

    def __init__(
        self,
        name,
        saveModelIn: str = "",
        verbose: bool = False,
        savelogDirectory: str = "",
    ):
        super().__init__(
            self.suffix,
            name,
            saveModelIn,
        )

        self.reward = RewardOnlyWinning()

        if verbose:
            self.startLogging(savelogDirectory)

    def get_action(self, observations):
        possibleActions = observations[28:]

        itemindex = numpy.array(numpy.where(numpy.array(possibleActions) == 1))[
            0
        ].tolist()

        if len(itemindex) > 1:
            itemindex.pop()

        aIndex = itemindex[-1:]
        a = numpy.zeros(200)
        a[aIndex] = 1

        return a

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def update_my_action(self, envInfo):  # aqui vem os dados depois da jogada
        pass

    def update_action_others(
        self, envInfo
    ):  # depois q o oponente fez uma acao, ele chata o action orders
        pass

    def update_end_match(self, envInfo):
        pass

    def update_start_match(self, cards, players, starting_player):
        pass

    def get_reward(self, envInfo):
        thisPlayer = envInfo["thisPlayerPosition"]
        matchFinished = envInfo["thisPlayerFinished"]

        # print(envInfo)

        return self.reward.getReward(thisPlayer, matchFinished)
