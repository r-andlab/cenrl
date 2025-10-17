import datetime
import ipaddress
import random
import typing

import pandas as pd

from common.utils import TARGET_FEATURE__SERVICE_IP, SERVER_ASN, SERVER_FEATURE_ORDER

NUM_BLOCKED_MEASUREMENTS = "num_blocked_measurements"
NUM_MEASUREMENTS = "num_measurements"

class TrieNodeIP:
    def __init__(self, num_blocked_measurements: int = 0,
                 num_measurements: int = 0):
        self.children = {}
        self.is_end_of_network = False
        self.num_blocked_measurements = num_blocked_measurements
        self.num_measurements = num_measurements


class IPBlocker:
    def __init__(self,
                 reward_probabilities: typing.Dict,
                 ip_addresses: typing.List[str] = None,
                 ip_addresses_measurements: typing.List[typing.Tuple[int, int]] = None,
                 ip_networks: typing.List[str] = None):

        self.root = TrieNodeIP()
        self.reward_probabilities = reward_probabilities
        if ip_addresses and ip_addresses_measurements:
            assert len(ip_addresses) == len(
                ip_addresses_measurements), "Length of ip_addresses and ip_addresses_measurements must be the same"
            for ip, measurement_tuple in zip(ip_addresses, ip_addresses_measurements):
                num_blocked_measurements, num_measurements = measurement_tuple
                self.insert_ip_address(ipaddress.ip_address(ip),
                                       num_blocked_measurements=num_blocked_measurements,
                                       num_measurements=num_measurements)
        elif ip_addresses:
            for ip in ip_addresses:
                self.insert_ip_address(ipaddress.ip_address(ip))

        if ip_networks:
            for net in ip_networks:
                self.insert_ip_network(net)

    def insert_ip_address(self, ip: typing.Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
                          num_blocked_measurements: int = 0,
                          num_measurements: int = 0
                          ):
        node = self.root
        bits = list(map(int, ip.packed))
        for bit in bits:
            if bit not in node.children:
                node.children[bit] = TrieNodeIP()
            node = node.children[bit]

        if node != self.root:
            node.is_end_of_network = True
            node.num_blocked_measurements = num_blocked_measurements
            node.num_measurements = num_measurements

    def insert_ip_network(self, network: str):
        for ip in ipaddress.ip_network(network).hosts():
            self.insert_ip_address(ip)

    def should_block(self, ip_address: str) -> typing.Tuple[bool, typing.Optional[TrieNodeIP]]:
        ip = list(map(int, ipaddress.ip_address(ip_address).packed))
        node = self.root
        for bit in ip:
            if bit not in node.children:
                break
            node = node.children[bit]
            if node.is_end_of_network:
                return True, node
        return False, None


def create_reward_probability_by_property(df: pd.DataFrame, property_name: str) -> dict:
    unique_prop_values = df[property_name].unique().tolist()

    reward_probabilities = dict()
    for prop_value in unique_prop_values:
        df_tmp = df[df[property_name] == prop_value]
        num_blocked_measurements = df_tmp[NUM_BLOCKED_MEASUREMENTS].sum()
        num_measurements = df_tmp[NUM_MEASUREMENTS].sum()
        ratio = 0
        if num_measurements > 0:
            ratio = round(float(num_blocked_measurements / num_measurements), 2)
        reward_probabilities[prop_value] = ratio

    return reward_probabilities


def create_reward_probabilities(df: pd.DataFrame) -> dict:
    reward_probabilities = dict()

    for feature in SERVER_FEATURE_ORDER:
        feature_probabilities = create_reward_probability_by_property(df, feature)
        reward_probabilities[feature] = feature_probabilities
    return reward_probabilities


def create_ipblocker_with_ground_truth(df: pd.DataFrame, target_feature: str) -> IPBlocker:
    assert target_feature == TARGET_FEATURE__SERVICE_IP, f"Expected {TARGET_FEATURE__SERVICE_IP} as target feature when building ipblocker"
    ip_addresses = df[target_feature].tolist()
    ip_addresses_measurements = [(x, y) for x, y in
                                 zip(df[NUM_BLOCKED_MEASUREMENTS].tolist(), df[NUM_MEASUREMENTS].tolist())]
    reward_probabilities = create_reward_probabilities(df)
    return IPBlocker(reward_probabilities,
                     ip_addresses=ip_addresses,
                     ip_addresses_measurements=ip_addresses_measurements)


def create_ipblocker_with_ground_truth_date(df: pd.DataFrame,
                                            target_feature: str,
                                            selected_date: datetime.datetime) -> dict[
    typing.Any, IPBlocker]:
    # turn domain into
    assert target_feature == TARGET_FEATURE__SERVICE_IP, f"Expected {TARGET_FEATURE__SERVICE_IP} as target feature when building ipblocker"
    blockers = dict()
    df_tmp = df[df["date"] == selected_date]
    ip_addresses = df_tmp[target_feature].tolist()
    ip_addresses_measurements = [(x, y) for x, y in
                                 zip(df_tmp[NUM_BLOCKED_MEASUREMENTS].tolist(), df_tmp[NUM_MEASUREMENTS].tolist())]

    reward_probabilities = create_reward_probabilities(df_tmp)
    blockers[selected_date] = IPBlocker(
        reward_probabilities,
        ip_addresses=ip_addresses,
        ip_addresses_measurements=ip_addresses_measurements)
    return blockers


def create_ipblocker_with_ground_truth_dates(df: pd.DataFrame, target_feature: str) -> dict[
    typing.Any, IPBlocker]:
    # turn domain into
    assert target_feature == TARGET_FEATURE__SERVICE_IP, f"Expected {TARGET_FEATURE__SERVICE_IP} as target feature when building ipblocker"
    blockers = dict()
    for d in df["date"].unique().tolist():
        df_tmp = df[df["date"] == d]
        ip_addresses = df_tmp[target_feature].tolist()
        ip_addresses_measurements = [(x, y) for x, y in zip(df_tmp[NUM_BLOCKED_MEASUREMENTS].tolist(),
                                                            df_tmp[NUM_MEASUREMENTS].tolist())]

        reward_probabilities = create_reward_probabilities(df_tmp)

        blockers[d] = IPBlocker(reward_probabilities,
                                ip_addresses=ip_addresses,
                                ip_addresses_measurements=ip_addresses_measurements)
    return blockers


def reward_in_blocklist_ipblocker(ipblocker: IPBlocker,
                                  ip_address: str,
                                  server_features: dict) -> typing.Tuple[float, bool]:
    is_in_blocklist, node = ipblocker.should_block(ip_address)

    reward = 0
    for feature in SERVER_FEATURE_ORDER:
        if feature in ipblocker.reward_probabilities and feature in server_features:
            feature_reward_probabilities = ipblocker.reward_probabilities[feature]
            feature_val = server_features[feature]
            if feature_val in feature_reward_probabilities:
                prob = feature_reward_probabilities[feature_val]
                if prob > 0:
                    #print(f"Found prob > 0: {prob} for {ip_address}, {feature}: {feature_val}")
                    if random.uniform(0, 1) <= prob:
                        reward = 1
                        # if reward is one, we treat it as is_in_blocklist
                        is_in_blocklist = True
                    break

    return reward, is_in_blocklist


def reward_in_blocklist_by_date_prioritize_changes_ipblocker(
        ipblocker: IPBlocker,
        ip_address: str,
        previous_results: list) -> typing.Tuple[float, bool]:

    # TODO/WARNING: Not sure how this is changed for IP using SERVER_ASN YET

    is_in_blocklist, _ = ipblocker.should_block(ip_address)

    reward = 0.5 if is_in_blocklist else 0

    if previous_results:
        prev_reward, prev_is_in_blocklist = previous_results[-1]
        if prev_is_in_blocklist != is_in_blocklist:
            reward += 0.5
            print(f"Found a change")

    return reward, is_in_blocklist
