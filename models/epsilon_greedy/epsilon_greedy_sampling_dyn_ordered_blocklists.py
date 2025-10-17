import datetime
import typing

from epsilon_greedy_sampling_dyn_blocklists import EpsilonGreedySamplingDynamicDate, \
    EpsilonGreedySamplingDynParserOptions
from models.base.action_space import NAME
from models.base.model import DynOrderedModelRun, run_multiprocessing
from common.utils import reward_in_blocklist_by_date
from models.base.utils_adblocker import reward_in_blocklist_adb, create_adblocker_with_ground_truth_date


class EpsilonGreedySamplingDynamicOrderedDate(EpsilonGreedySamplingDynamicDate):
    """
        This class takes in the ground truth file and splits it up by date for multiple blocklists
        At each take_measurements, we will select a blocklist based on the current date.
        Advance the date after per_date_threshold number of iterations
        """

    def __init__(self, params,
                 per_date_threshold: int = 2000,
                 reset_more_per_date_threshold: bool = False,
                 **kwargs):
        self.per_date_threshold = per_date_threshold
        self.reset_more_per_date_threshold = reset_more_per_date_threshold
        self._current_date = None
        self.selected_target_measurement_cache = dict()
        super().__init__(params, **kwargs)

    def reset(self):
        super().reset()
        self.selected_target_measurement_cache.clear()
        self._current_date = None
        self.parse_block_list()
        self.init_blockers()

    def init_blockers(self):
        self.blockers.clear()
        self.blockers.update(create_adblocker_with_ground_truth_date(self.blocklist,
                                                                     self.target_feature,
                                                                     self._current_date))

    def set_blocklist_unique_counts_based_on_action_space(self):
        """
        Run the action space target nodes on the adblocker to find the max number of items we can find that are blocked
        We have to eat the cost once
        """
        for key, adb in self.blockers.items():
            block_count = 0
            for n, n_data in self.action_space.gen_active_target_nodes_and_data():
                target_name = n_data[NAME]
                _, is_blocked = reward_in_blocklist_adb(adb, target_name)
                if is_blocked:
                    block_count += 1
            self.blocklist_unique_count[key] = block_count
            self.blocklist_targets_found[key] = dict()

        #for key in self.blocklist_unique_count:
        #    match_df = self.blocklist[self.blocklist["date"] == key]
        #    if len(match_df) > 0:
        #        print(
        #            f"{key}: diff in unique blocklist count {self.blocklist_unique_count[key]}, vs. all {len(match_df[self.target_feature].unique())}")

    def update_blocklist_target_found(self, target_found: str):
        if self._current_date not in self.blocklist_targets_found:
            self.set_blocklist_unique_counts_based_on_action_space()
        self.blocklist_targets_found[self._current_date][target_found] = 1

    def get_blocklist_coverage(self) -> float:
        if self._current_date not in self.blocklist_unique_count:
            self.set_blocklist_unique_counts_based_on_action_space()
        return round(len(self.blocklist_targets_found[self._current_date]) / float(self.blocklist_unique_count[self._current_date]), 2)

    def parse_block_list(self):
        super().parse_block_list()
        self.set_next_blocklist_date()

    def set_next_blocklist_date(self):
        dates = self.blocklist["date"].unique().tolist()
        dates.sort()
        if self._current_date is None:
            self._current_date = dates[0]
            # print(f"prev date None, next date: {dates[0]}")
        else:
            for d in dates:
                if d > self._current_date:
                    # print(f"prev date {self._current_date}, next date: {d}")
                    self._current_date = d
                    break

    def step(self):
        step_results = super().step()
        step_results["date"] = self._current_date
        # if we reach the threshold, go to next date and clear disabled arms/targets
        if self.current_epoch_num > 0 and \
                self.current_epoch_num % self.per_date_threshold == 0:
            self.set_next_blocklist_date()
            self.init_blockers()
            self.action_space.wake_up_all_nodes()
            self.selected_target_count.clear()
            if self.reset_more_per_date_threshold:
                print(f"reset_more_per_date_threshold = True")
                self.action_space.reset_action_attempts()
                self.optimal_value = 0
                self.last_selected_arm_index = None
                # reset internal trackers
                self._selected_arm_key = None
                self._selected_arm_name = None
                self._selected_target = None
                self._selected_target_name = None
        return step_results

    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        """
        Randomly selects a blocklist by date, then see if the selected_target is in that blocklist
        """
        return reward_in_blocklist_adb(self.blockers[self._current_date], selected_target)


class EpsilonGreedySamplingOrderedParserOptions(EpsilonGreedySamplingDynParserOptions):
  def add_arguments(self):
    super().add_arguments()
    self.parser.add_argument("-pdt", "--per_date_threshold", type=int, default=2000)
    self.parser.add_argument("-mpdt", "--more_per_date_threshold", type=bool, default=False)
    self.parser.add_argument("-rmpdt", "--reset_more_per_date_threshold", action="store_true")

  def set_params(self, args):
    super().set_params(args)
    self.params["per_date_threshold"] = args.per_date_threshold
    self.params["reset_more_per_date_threshold"] = args.reset_more_per_date_threshold


if __name__ == "__main__":
    parser = EpsilonGreedySamplingOrderedParserOptions()
    params = parser.parse()

    addition_kwargs = {"selected_target_max_try": params["selected_target_max_try"],
                       "per_date_threshold": params["per_date_threshold"],
                       "reset_more_per_date_threshold": params["reset_more_per_date_threshold"],
                       "model_runner": DynOrderedModelRun()}

    run_multiprocessing(EpsilonGreedySamplingDynamicOrderedDate, params, addition_kwargs=addition_kwargs)
