import datetime
import random
import typing

import numpy as np
import pandas as pd

from models.base.model import DynModelRun, run_multiprocessing
from common.utils import reward_in_blocklist_by_date, NO_DATE_BLOCKLIST
from models.base.dyn_env_utils import disable_target, take_measurements_random_date
from models.base.utils_adblocker import create_adblocker_with_ground_truth_dates
from ucb_naive import UCBNaive, UCBNaiveParserOptions

np.seterr(divide="ignore", invalid="ignore")


class UCBNaiveDynamicDate(UCBNaive):
    """
    This class takes in the ground truth file and splits it up by date for multiple blocklists
    At each take_measurements, we will randomly select a blocklist
    """

    def __init__(self, params, selected_target_max_try: int = 10, **kwargs):
        super().__init__(params, **kwargs)
        self.selected_target_count = dict()
        self.selected_target_max_try = selected_target_max_try
        self.last_selected_date = NO_DATE_BLOCKLIST
        # print(f"using selected_target_max_try {self.selected_target_max_try}")

    def reset(self):
        super().reset()
        self.selected_target_count.clear()

    def disable_target(self, target: str):
        disable_target(target, self.selected_target_count, self.selected_target_max_try, self.action_space)

    def parse_block_list(self):
        super().parse_block_list()
        self.blocklist["date"] = self.blocklist["date"].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))

    def get_blocklist_coverage(self) -> float:
        coverage = 0
        if self.last_selected_date in self.blocklist_targets_found and self.last_selected_date in self.blocklist_unique_count:
            coverage = round(len(self.blocklist_targets_found[self.last_selected_date]) / float(self.blocklist_unique_count[
                                                                                              self.last_selected_date]), 2)

        return coverage if coverage <= 1 else 1

    def update_blocklist_target_found(self, target_found: str):
        if self.last_selected_date and self.last_selected_date in self.blocklist_targets_found:
            self.blocklist_targets_found[self.last_selected_date][target_found] = 1

    def init_blockers(self):
        self.blockers.update(create_adblocker_with_ground_truth_dates(self.blocklist, self.target_feature))

    # the actual measurement-taking part of the original make_measurement function.
    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        """
        Randomly selects a blocklist by date, then see if the selected_target is in that blocklist
        """
        reward, is_in_blocklist, selected_date = take_measurements_random_date(self.blocklist, self.blockers, selected_target, self.target_feature)
        self.last_selected_date = selected_date

        return reward, is_in_blocklist

class UCBNaiveDynParserOptions(UCBNaiveParserOptions):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument("-stmt", "--selected_target_max_try",
                                 type=int, default=10,
                                 help="number of max tries per item selected for measurement (like domain)")

    def set_params(self, args):
        super().set_params(args)
        self.params["selected_target_max_try"] = args.selected_target_max_try


if __name__ == "__main__":
    parser = UCBNaiveDynParserOptions()
    params = parser.parse()

    addition_kwargs = {"selected_target_max_try": params["selected_target_max_try"],
                       "model_runner": DynModelRun()}

    run_multiprocessing(UCBNaiveDynamicDate, params, addition_kwargs=addition_kwargs)

if __name__ == "__main__":
    parser = UCBNaiveParserOptions()
    params = parser.parse()
    run_multiprocessing(UCBNaive, params)
