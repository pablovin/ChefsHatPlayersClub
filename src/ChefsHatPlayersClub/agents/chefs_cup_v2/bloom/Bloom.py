from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
import numpy
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning


class Bloom(ChefsHatPlayer):
    suffix = "BLOOM"

    def __init__(
        self,
        name,
        saveFolder: str = "",
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
        self.reward = RewardOnlyWinning()

    # Receives action
    def get_action(self, observations):
        possibleActions = observations[28:]
        itemindex = numpy.array(numpy.where(numpy.array(possibleActions) == 1))[
            0
        ].tolist()
        aIndex = numpy.max(itemindex[:-1]) if len(itemindex) > 1 else itemindex[0]
        a = numpy.zeros(200)
        a[aIndex] = 1
        return a

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards)[-amount:]
        return selectedCards

    def do_special_action(self, info, specialAction):
        return True

    def update_my_action(self, envInfo):
        pass

    # Observes the game
    def update_action_others(self, envInfo):
        pass

    def update_end_match(self, envInfo):
        pass

    def update_start_match(self, cards, players, starting_player):
        pass

    def get_reward(self, envInfo):
        this_player = envInfo["Author_Index"]
        this_player_position = 3 - envInfo["Match_Score"][this_player]
        this_player_finished = envInfo["Finished_Players"][this_player]

        return self.reward.getReward(this_player_position, this_player_finished)
