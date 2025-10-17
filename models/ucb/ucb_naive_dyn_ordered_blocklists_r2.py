import typing

import numpy as np

from models.base.action_space import NAME
from models.base.dyn_env_utils import take_measurement_date_r2
from models.base.model import DynOrderedModelRun, run_multiprocessing
from models.base.utils_adblocker import reward_in_blocklist_by_date_prioritize_changes_adb
from ucb_naive_dyn_ordered_blocklists import UCBNaiveDynamicOrderedDate, UCBNaiveOrderedParserOptions

np.seterr(divide="ignore", invalid="ignore")


class UCBNaiveDynamicOrderedDateR2(UCBNaiveDynamicOrderedDate):

    def set_blocklist_unique_counts_based_on_action_space(self):
        """
        Run the action space target nodes on the adblocker to find the max number of items we can find that are blocked
        We have to eat the cost once
        """
        for key, adb in self.blockers.items():
            block_count = 0
            for n, n_data in self.action_space.gen_active_target_nodes_and_data():
                target_name = n_data[NAME]
                _, is_blocked = reward_in_blocklist_by_date_prioritize_changes_adb(adb,
                                                                                   target_name,
                                                                                   self.selected_target_measurement_cache.get(
                                                                                       target_name, []))

                if is_blocked:
                    block_count += 1

            self.blocklist_unique_count[key] = block_count
            self.blocklist_targets_found[key] = dict()

        #for key in self.blocklist_unique_count:
        #    match_df = self.blocklist[self.blocklist["date"] == key]
        #    if len(match_df) > 0:
        #        print(
        #            f"{key}: diff in unique blocklist count {self.blocklist_unique_count[key]}, vs. all {len(match_df[self.target_feature].unique())}")

    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        return take_measurement_date_r2(self.blockers,
                                        selected_target,
                                        self._current_date,
                                        self.selected_target_measurement_cache)


if __name__ == "__main__":
    parser = UCBNaiveOrderedParserOptions()
    params = parser.parse()

    addition_kwargs = {"selected_target_max_try": params["selected_target_max_try"],
                       "per_date_threshold": params["per_date_threshold"],
                       "reset_more_per_date_threshold": params["reset_more_per_date_threshold"],
                       "model_runner": DynOrderedModelRun()}

    run_multiprocessing(UCBNaiveDynamicOrderedDateR2, params, addition_kwargs=addition_kwargs)
