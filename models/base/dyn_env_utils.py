import datetime
import random
import typing

import braveblock
import pandas as pd

from models.base.action_space import ActionSpaceBase
from models.base.utils_adblocker import reward_in_blocklist_adb, reward_in_blocklist_by_date_prioritize_changes_adb


def disable_target(target: str,
                   selected_target_count: dict,
                   selected_target_max_try: int,
                   action_space: ActionSpaceBase) -> typing.List[str]:
    if target not in selected_target_count:
        selected_target_count[target] = 0
    selected_target_count[target] += 1
    if selected_target_count[target] >= selected_target_max_try:
        return action_space.put_to_sleep(target)


def take_measurements_random_date(blocklist: pd.DataFrame, adblockers: typing.List[braveblock.Adblocker],
                                  selected_target: str, target_feature: str) -> typing.Tuple[float, bool, datetime.datetime]:
    selected_date = random.choice(blocklist["date"].unique().tolist())
    reward, is_in_blocklist = reward_in_blocklist_adb(adblockers[selected_date], selected_target)
    return reward, is_in_blocklist, selected_date


def take_measurement_date_r2(adblockers: typing.List[braveblock.Adblocker],
                             selected_target: str,
                             selected_date: datetime,
                             selected_target_measurement_cache: dict) -> typing.Tuple[float, bool]:
    """
    Randomly selects a blocklist by date, then see if the selected_target is in that blocklist
    """
    reward, is_in_blocklist = reward_in_blocklist_by_date_prioritize_changes_adb(adblockers[selected_date],
                                                                                 selected_target,
                                                                                 selected_target_measurement_cache.get(
                                                                                     selected_target, []))

    if selected_target not in selected_target_measurement_cache:
        selected_target_measurement_cache[selected_target] = []

    selected_target_measurement_cache[selected_target].append((reward, is_in_blocklist))

    return reward, is_in_blocklist
