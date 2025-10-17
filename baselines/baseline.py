import typing

import braveblock
import pandas as pd

from baselines.utils import COLUMN_NAME_RANK
from common.utils import NO_DATE_BLOCKLIST, TARGET_FEATURE__DOMAIN, TARGET_FEATURE__SERVICE_IP, \
    SERVER_FEATURE_ORDER
from models.base.utils_adblocker import reward_in_blocklist_adb
from models.base.utils_ipblocker import IPBlocker, reward_in_blocklist_ipblocker


class Baseline:
    def __init__(self,
                 action_space_path: str,
                 blockers: dict,
                 episode: int = 0,
                 target_feature: str = TARGET_FEATURE__DOMAIN,
                 unique_dates: list = None,
                 with_replacement: bool = True,
                 max_entry_retry: int = 1,
                 measurements: int = -1):
        """
            Args:
                action_space_path: path to action space list
                blockers: loaded adblocker/ipblocker with ground truth blocklist
                with_replacement: if True, don't remove the entry selection
                max_entry_retry: how many times we can select an entry before removal (used when with_replacement=True)
                measurements: overall number of measurements (this affects the return list)

        """
        self.target_feature = target_feature
        self.action_space_path = action_space_path
        self.blockers = blockers
        self.unique_dates = unique_dates
        self.with_replacement = with_replacement
        self.max_entry_retry = max_entry_retry
        self.measurements = measurements
        self.episode = episode
        self.blocklist_unique_count = dict()
        self.blocklist_targets_found = dict()
        self.results = []

        self.df_action_space_file = pd.read_csv(self.action_space_path, delimiter="|", index_col=False)

        # Uncomment out below to test sample a smaller action space file
        #SAMPLE_ACTION_SPACE = 10000
        #if len(self.df_action_space_file) > SAMPLE_ACTION_SPACE:
        #    self.df_action_space_file = self.df_action_space_file.sample(SAMPLE_ACTION_SPACE, random_state=40)
        #    print(f"Warning: Using only {SAMPLE_ACTION_SPACE} rows of the action space")

        if COLUMN_NAME_RANK in self.df_action_space_file.columns:
            self.df_action_space_file = self.df_action_space_file.sort_values(by=[COLUMN_NAME_RANK], ascending=True)
        self.set_blocklist_unique_counts_based_on_action_space()

    def update_blocklist_target_found(self, target_found: str):
        self.blocklist_targets_found[NO_DATE_BLOCKLIST][target_found] = 1

    def get_blocklist_coverage(self) -> float:
        coverage = round(len(self.blocklist_targets_found[NO_DATE_BLOCKLIST]) / float(
            self.blocklist_unique_count[NO_DATE_BLOCKLIST]), 2)

        return coverage if coverage <= 1 else 1

    def reward_in_blocklist(self, blocker: typing.Union[braveblock.Adblocker, IPBlocker], selected_target: str) -> typing.Tuple[float, bool]:
        if self.target_feature == TARGET_FEATURE__DOMAIN:
            return reward_in_blocklist_adb(blocker, selected_target)
        if self.target_feature == TARGET_FEATURE__SERVICE_IP:
            server_features = dict()
            for feature in SERVER_FEATURE_ORDER:
                if feature in self.df_action_space_file.columns:
                    feature_val = \
                        self.df_action_space_file[self.df_action_space_file[self.target_feature] == selected_target][
                        feature]
                    if len(feature_val) > 0:
                        feature_val = feature_val.iloc[0]
                        server_features[feature] = feature_val
            return reward_in_blocklist_ipblocker(blocker, selected_target, server_features)

    def set_blocklist_unique_counts_based_on_action_space(self):
        """
        Run the action space target nodes on the adblocker to find the max number of items we can find that are blocked
        We have to eat the cost once
        """
        entries_in_order = self.df_action_space_file[self.target_feature].to_list()

        for key, blocker in self.blockers.items():
            block_count = 0
            for target_name in entries_in_order:
                _, is_blocked = self.reward_in_blocklist(blocker, target_name)
                if is_blocked:
                    block_count += 1
            self.blocklist_unique_count[key] = block_count
            self.blocklist_targets_found[key] = dict()

    def take_action(self,
                    selected_target: str = None,
                    broader_action: str = None,
                    blocker_key: typing.Any = NO_DATE_BLOCKLIST,
                    ):

        is_blocked = False
        reward = 0
        if selected_target:
            reward, is_blocked = self.reward_in_blocklist(self.blockers[blocker_key], selected_target)
            if is_blocked:
                self.update_blocklist_target_found(selected_target)

        return {
            "episode": self.episode,
            "action": broader_action or selected_target or "",
            "target": selected_target or "",
            "reward": round(reward, 2),
            "is_blocked": 1 if is_blocked else 0,
            "coverage": self.get_blocklist_coverage()
        }

    def run(self) -> list:
        raise NotImplemented()


class BaselineWithDate (Baseline):
    """
    Baselines that consider dates
    """

    def __init__(self, *args,
                 per_date_threshold: int = 2000,
                 use_ordered_dates: bool = False,
                 **kwargs):
        self.per_date_threshold = per_date_threshold
        self.use_ordered_dates = use_ordered_dates
        self.current_date = None
        super().__init__(*args, **kwargs)

    def update_blocklist_target_found(self, target_found: str):
        if self.current_date not in self.blocklist_targets_found:
            self.set_blocklist_unique_counts_based_on_action_space()
        self.blocklist_targets_found[self.current_date][target_found] = 1

    def get_blocklist_coverage(self) -> float:
        if self.current_date not in self.blocklist_unique_count:
            self.set_blocklist_unique_counts_based_on_action_space()
        return round(len(self.blocklist_targets_found[self.current_date]) / float(self.blocklist_unique_count[self.current_date]), 2)

