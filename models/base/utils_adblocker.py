import datetime
import math
import typing

import braveblock
import pandas as pd

from common.utils import TARGET_FEATURE__DOMAIN, HTTPS_PROTOCOL


def create_adblocker_with_ground_truth(df: pd.DataFrame, target_feature: str) -> braveblock.Adblocker:
    # turn domain into
    assert target_feature == TARGET_FEATURE__DOMAIN, f"Expected {TARGET_FEATURE__DOMAIN} as target feature when building adblocker"
    rules = [f"||{x}^" for x in df[target_feature].unique().tolist()]
    return braveblock.Adblocker(rules=rules,
                                include_easylist=False,
                                include_easyprivacy=False)


def create_adblocker_with_ground_truth_date(df: pd.DataFrame,
                                            target_feature: str,
                                            selected_date: datetime.datetime) -> dict[
    typing.Any, braveblock.Adblocker]:
    # turn domain into
    assert target_feature == TARGET_FEATURE__DOMAIN, f"Expected {TARGET_FEATURE__DOMAIN} as target feature when building adblocker"
    adblockers = dict()
    df_tmp = df[df["date"] == selected_date]
    rules = [f"||{x}^" for x in df_tmp[target_feature].unique().tolist()]
    #print(f"number of rules {len(rules)} for date {selected_date}")
    adblockers[selected_date] = braveblock.Adblocker(rules=rules,
                                         include_easylist=False,
                                         include_easyprivacy=False)
    return adblockers


def create_adblocker_with_ground_truth_dates(df: pd.DataFrame, target_feature: str) -> dict[
    typing.Any, braveblock.Adblocker]:
    # turn domain into
    assert target_feature == TARGET_FEATURE__DOMAIN, f"Expected {TARGET_FEATURE__DOMAIN} as target feature when building adblocker"
    adblockers = dict()
    for d in df["date"].unique().tolist():
        df_tmp = df[df["date"] == d]
        rules = [f"||{x}^" for x in df_tmp[target_feature].unique().tolist()]
        adblockers[d] = braveblock.Adblocker(rules=rules,
                                             include_easylist=False,
                                             include_easyprivacy=False)
    return adblockers


def is_blocked_by_adblocker(blocker: braveblock.Adblocker, selected_target: str) -> bool:
    return blocker.check_network_urls(
        url=f"{HTTPS_PROTOCOL}{selected_target}",
        source_url=f"{HTTPS_PROTOCOL}{selected_target}",
        request_type="other",
    )


def reward_in_blocklist_adb(adblocker: braveblock.Adblocker, selected_target: str) -> typing.Tuple[float, bool]:
    is_in_blocklist = is_blocked_by_adblocker(adblocker, selected_target)
    return 1 if is_in_blocklist else 0, is_in_blocklist


def reward_in_blocklist_by_date_prioritize_changes_adb(
                                adblocker: braveblock.Adblocker,
                                selected_target: str,
                                previous_results: list) -> typing.Tuple[float, bool]:

    is_in_blocklist = is_blocked_by_adblocker(adblocker, selected_target)
    reward = 0.5 if is_in_blocklist else 0

    if previous_results:
        prev_reward, prev_is_in_blocklist = previous_results[-1]
        if prev_is_in_blocklist != is_in_blocklist:
            reward += 0.5
            print(f"Found a change")

    return reward, is_in_blocklist


def reward_in_blocklist_adb_diverse(adblocker: braveblock.Adblocker,
                                    feature: str,
                                    selected_target: str,
                                    prev_blocked_by_feature: dict = None,
                                    decay_factor: float = 2) -> typing.Tuple[float, bool, dict]:


    reward, is_in_blocklist = reward_in_blocklist_adb(adblocker, selected_target)

    # add rarity value if is_in_blocklist=True
    rarity = 0
    if is_in_blocklist and feature and prev_blocked_by_feature is not None:
        total_blocked = sum(list(prev_blocked_by_feature.values()))

        if feature not in prev_blocked_by_feature:
            prev_blocked_by_feature[feature] = 0

        current_feature_blocked = prev_blocked_by_feature.get(feature)

        # rarity is calculated as the 1 - (the proportion of found blocked for that feature over the total blocked found)
        # this means that for a feature that has alot of blocked found, e.g. 9/10, then its rarity would be 1 - 0.9 = 0.1 (not that rare).

        if total_blocked > 0:
            rarity = 1 - math.pow(current_feature_blocked/total_blocked, decay_factor)
            #print(f"Rarity {rarity}, decay factor {decay_factor} - for feature {feature} and selected target {selected_target}, feature has been found blocked: {current_feature_blocked} out of {total_blocked}")

        # increment blocked count for that feature
        prev_blocked_by_feature[feature] += 1

    return reward + rarity, is_in_blocklist, prev_blocked_by_feature
