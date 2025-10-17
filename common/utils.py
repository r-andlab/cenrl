import re
import typing

import numpy as np
import pandas as pd

LOG_FILE_DELIMITER = "|"
NO_DATE_BLOCKLIST = "NO_DATE_BLOCKLIST"
TARGET_FEATURE__DOMAIN = "domain"
TARGET_FEATURE__SERVICE_IP = "server_ip"
FEATURE_CATEGORIES = "categories"
FEATURE_VPS = "vps"
HTTPS_PROTOCOL = "https://"
UNKNOWN_EMPTY = "Empty"
SERVER_ASN = "server_asn"
SERVER_NETBLOCK = "server_netblock"
SERVER_ORG = "server_organization"
SERVER_COUNTRY = "server_country"

SERVER_FEATURE_ORDER = [SERVER_NETBLOCK, SERVER_ORG, SERVER_ASN, SERVER_COUNTRY]

def get_regex_column_name(prefix: str) -> str:
    return prefix + "_regex"


def get_cumulative_avg_across_episodes(df: pd.DataFrame,
                                       property_column_name: str,
                                       episode_column_name: str = "episode") -> np.ndarray:

    cumsum_eps = []
    episodes = df[episode_column_name].unique().tolist()
    for ep in episodes:
        r = np.cumsum(df[df[episode_column_name] == ep][property_column_name].tolist())
        cumsum_eps.append(r)

    #print(cumsum_eps)
    return np.mean(cumsum_eps, axis=0)


def get_avg_across_episodes(df: pd.DataFrame, property_column_name: str, episode_column_name: str = "episode") -> np.ndarray:
    ep_values = []
    episodes = df[episode_column_name].unique().tolist()
    for ep in episodes:
        r = df[df[episode_column_name] == ep][property_column_name].tolist()
        ep_values.append(r)

    return np.mean(ep_values, axis=0)


def reward_in_blocklist(blocklist: typing.Any, selected_target: str, target_feature: str) -> int:
    if isinstance(blocklist, pd.DataFrame):
        if target_feature == TARGET_FEATURE__DOMAIN:
            regex_column = get_regex_column_name (target_feature)
            matches = blocklist[blocklist[regex_column].apply(lambda x: re.match(x, selected_target) is not None)]
            #print(f"{selected_target}: regex matches {matches}")
            is_in_blocklist = len(matches) > 0
        else:
            is_in_blocklist = len(blocklist[blocklist[target_feature] == selected_target]) > 0
        return 1 if is_in_blocklist else 0
    elif isinstance(blocklist, list):
        if target_feature == TARGET_FEATURE__DOMAIN:
            blocklist = [".*" + x.replace('.', r'\.').replace('*', '.*') for x in blocklist]
            is_in_blocklist = any((re.match(pattern, selected_target) for pattern in blocklist))
            return 1 if is_in_blocklist else 0
        else:
            return 1 if selected_target in blocklist else 0

    raise Exception(f"Cannot calculate reward due to unexpected blocklist type {blocklist}")


def reward_in_blocklist_by_date(blocklist_df: pd.DataFrame,
                                selected_target: str,
                                selected_dated: str,
                                target_feature: str,
                                date_column_name: str = "date") -> int:
    blocklist_df[target_feature] = blocklist_df[target_feature].str.replace('.', r'\.').str.replace('*', '.*')
    blocklist_df[target_feature] = '.*' + blocklist_df[target_feature]
    is_in_blocklist = len(blocklist_df[blocklist_df[target_feature].apply(lambda x: re.match(x, selected_target) is not None) & (blocklist_df[target_feature] == selected_target)]) > 0
    return 1 if is_in_blocklist else 0


def reward_in_blocklist_by_date_prioritize_changes(
                                blocklist_df: pd.DataFrame,
                                selected_target: str,
                                selected_dated: str,
                                target_feature: str,
                                previous_results: list,
                                date_column_name: str = "date") -> typing.Tuple[float, bool]:

    is_in_blocklist = len(blocklist_df[blocklist_df[target_feature].apply(lambda x: re.match(x, selected_target) is not None)  & (blocklist_df[target_feature] == selected_target)]) > 0

    reward = 0.5 if is_in_blocklist else 0

    if previous_results:
        prev_reward, prev_is_in_blocklist = previous_results[-1]
        if prev_is_in_blocklist != is_in_blocklist:
            reward += 0.5
            print(f"Found a change")

    return reward, is_in_blocklist


