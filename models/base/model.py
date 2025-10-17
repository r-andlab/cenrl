# Base class for models.
import collections
import os
import sys
import typing
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import braveblock
import pandas as pd

import models.base.action_space as action_space_module
from common.utils import LOG_FILE_DELIMITER, NO_DATE_BLOCKLIST, TARGET_FEATURE__DOMAIN, UNKNOWN_EMPTY, \
    TARGET_FEATURE__SERVICE_IP, SERVER_FEATURE_ORDER
from models.base.preprocessor import run_preprocessor
from models.base.utils_adblocker import create_adblocker_with_ground_truth, reward_in_blocklist_adb
from models.base.utils_ipblocker import create_ipblocker_with_ground_truth, reward_in_blocklist_ipblocker, IPBlocker


class Model:
    def __init__(self, params,
                 model_runner: "ModelRunBase" = None):
        self.blocklist = None
        self.target_feature = params["target_feature"]
        self.verbose = params["verbose"]
        self.output_directory = params["output_directory"]
        self.outfile = params["outfile_csv"]
        #self.logfile = params["logfile"]
        self.logfile = None
        self.ground_truth_path = params["ground_truth_path"]
        self.action_space_file = params["action_space_file"]
        self.action_value_file = params["action_value_file"]
        self.features = params["features"]
        self.consider_unknown = params["consider_unknown"]
        self.sample_by_target_rank = params["sample_by_target_rank"]
        self.last_selected_arm_index = None
        self.model_runner = model_runner
        self.action_space: typing.Optional[action_space_module.ActionSpaceBase] = None

        if self.model_runner is None:
            self.model_runner = ModelRunBase()

        self.num_episodes = params["num_episodes"]
        self.measurements_per_episode = params["measurements_per_episode"]
        self.action_space_multi_parents = params["action_space_multi_parents"]

        self.current_epoch_num = 0
        self.optimal_value = 0
        self.blocklist_unique_count = dict()
        self.blocklist_targets_found = dict()
        self.blockers = dict()

        print(f"Episode Process {os.getpid()} - Parsing blocklist and init blockers")
        self.parse_block_list()
        self.init_blockers()
        print(f"Episode Process {os.getpid()} - Done: Parsing blocklist and init blockers")

        # keep track of selection of arms (feature) and targets (target_feature)
        self._selected_arm_key = None
        self._selected_arm_name = None
        self._selected_target = None
        self._selected_target_name = None

        print(f"Episode Process {os.getpid()} - model is using target feature {self.target_feature}")

    def reset(self):
        # This value is incremneted in run() in the innermost for loop:
        #     for self.current_epoch_num in tqdm(range(self.measurements_per_episode)):
        self.current_epoch_num = 0
        self.optimal_value = 0
        self.last_selected_arm_index = None
        for key, targets_dict in self.blocklist_targets_found.items():
            targets_dict.clear()
        if self.action_space:
            self.action_space.reset()

        # reset internal trackers
        self._selected_arm_key = None
        self._selected_arm_name = None
        self._selected_target = None
        self._selected_target_name = None

    def save(self):
        self.action_space.save()

    def choose_arm(self) -> typing.List[str]:
        """
        Returns: a sequence of arms chosen if the action space is hierarchical

        """
        raise NotImplementedError

    def choose_target(self, selected_arm_key: str, selected_arm_name: str) -> str:
        chosen_targets = self.action_space.sample_successors(selected_arm_key,
                                                             use_rank_weights=self.sample_by_target_rank)
        return chosen_targets[0]

    def set_blocklist_unique_counts_based_on_action_space(self):
        """
        Run the action space target nodes on the adblocker to find the max number of items we can find that are blocked
        We have to eat the cost once
        """
        for key, blocker in self.blockers.items():
            block_count = 0
            for n, n_data in self.action_space.gen_active_target_nodes_and_data():
                target_name = n_data[action_space_module.NAME]
                _, is_blocked = self.take_measurement_by_blocker(blocker, target_name)
                if is_blocked:
                    block_count += 1
            self.blocklist_unique_count[key] = block_count
            self.blocklist_targets_found[key] = dict()

        # print(f"diff in unique blocklist count {self.blocklist_unique_count[NO_DATE_BLOCKLIST]}, vs. all {len(self.blocklist[self.target_feature].unique())}")

    def update_blocklist_target_found(self, target_found: str):
        self.blocklist_targets_found[NO_DATE_BLOCKLIST][target_found] = 1

    def get_blocklist_coverage(self) -> float:
        coverage = round(len(self.blocklist_targets_found[NO_DATE_BLOCKLIST]) / float(self.blocklist_unique_count[
                                                                                      NO_DATE_BLOCKLIST]), 2)
        return coverage if coverage <= 1 else 1

    def parse_block_list(self):
        df = pd.read_csv(self.ground_truth_path, delimiter=LOG_FILE_DELIMITER)
        if self.target_feature not in df.columns:
            print(f"Episode Process {os.getpid()} - warning, reading in blocklist does not have expected columns with delimiter {LOG_FILE_DELIMITER}")
            df = pd.read_csv(self.ground_truth_path, delimiter=",")
        assert self.target_feature in df.columns, \
            f"Ground truth file {self.ground_truth_path} does not have target feature {self.target_feature}"

        self.blocklist = df

    def init_blockers(self):
        if self.target_feature == TARGET_FEATURE__DOMAIN:
            self.blockers[NO_DATE_BLOCKLIST] = create_adblocker_with_ground_truth(self.blocklist, self.target_feature)
        if self.target_feature == TARGET_FEATURE__SERVICE_IP:
            self.blockers[NO_DATE_BLOCKLIST] = create_ipblocker_with_ground_truth(self.blocklist, self.target_feature)

    # the actual measurement-taking part of the original make_measurement function.
    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        return self.take_measurement_by_blocker(self.blockers[NO_DATE_BLOCKLIST], selected_target)

    def take_measurement_by_blocker(self, blocker: typing.Union[braveblock.Adblocker, IPBlocker], selected_target: str) -> typing.Tuple[float, bool]:
        if self.target_feature == TARGET_FEATURE__DOMAIN:
            return reward_in_blocklist_adb(blocker, selected_target)
        if self.target_feature == TARGET_FEATURE__SERVICE_IP:
            server_features = dict()
            n_data = self.action_space.get_by_property(action_space_module.NAME, selected_target)
            for feature in SERVER_FEATURE_ORDER:
                if feature in n_data:
                    server_features[feature] = n_data[feature]
            return reward_in_blocklist_ipblocker(blocker, selected_target, server_features)

    # the updates to self variables.
    def observe(self, selected_arm: str, measurement_result: float) -> float:
        """Needs to return the observed value"""
        raise NotImplementedError

    def propagate_rewards(self, selected_arm: str):
        d = collections.deque()
        d.append(selected_arm)

        while d:
            curr_node = d.popleft()

            # ignore root node
            if curr_node == self.action_space.get_root():
                break

            # if it does not only have target successors, then calculate its aggregated q_value
            if not self.action_space.has_target_successors(curr_node):
                successors = self.action_space.get_active_nontarget_successors(curr_node)
                if successors:
                    q_values = [self.action_space.get(x)[action_space_module.Q_VALUE] for x in successors]
                    agg_q_value = round(mean(q_values), 2)
                    self.action_space.get(curr_node)[action_space_module.Q_VALUE] = agg_q_value
                    # print(f"agg value {agg_q_value} for {curr_node}")

            # add parents
            for p in self.action_space.get_graph().predecessors(curr_node):
                d.append(p)

    def disable_target(self, target: str) -> typing.List[str]:
        return self.action_space.put_to_sleep(target)

    def can_step(self) -> bool:
        return self.action_space.has_active_nontarget_node()

    def is_optimal_action(self, selected_arm_seq: typing.List[str]) -> bool:
        selected_arm_seq_value = round(
            sum([self.action_space.get(n)[action_space_module.Q_VALUE] for n in selected_arm_seq]), 2)
        if selected_arm_seq_value >= self.optimal_value:
            self.optimal_value = selected_arm_seq_value
            return True
        return False

    def update_optimal_value(self):
        """
        At each layer of action space, find the highest Q_VALUE, then sum it as the optimal_value
        """
        root = self.action_space.get_root()
        d = collections.deque()
        d.append(root)
        value_seq = []
        while d:
            n = d.popleft()
            if self.action_space.has_target_successors(n):
                break

            max_value = 0
            max_node = None
            for succ in self.action_space.get_active_nontarget_successors(n):
                succ_value = self.action_space.get(succ)[action_space_module.Q_VALUE]
                if succ_value > max_value:
                    max_value = succ_value
                    max_node = succ

            value_seq.append(max_value)
            if max_node:
                d.append(max_node)

        self.optimal_value = round(sum(value_seq), 2)

    def step(self) -> dict:
        selected_arm_seq = self.choose_arm()
        self._selected_arm_key = selected_arm_seq[-1]

        self._selected_arm_name = self.action_space.get(self._selected_arm_key)[action_space_module.NAME]
        # print(f"selected arm: index: {selected_arm_index}, name: {selected_arm_name}")
        if self.verbose and self.logfile:
            self.logfile.write(str(self._selected_arm_name) + LOG_FILE_DELIMITER)

        self._selected_target = self.choose_target(self._selected_arm_key, self._selected_arm_name)
        self._selected_target_name = self.action_space.get(self._selected_target)[action_space_module.NAME]

        # NOTE: we need to pass in the NAME of the node because thats what the target_feature raw data is.
        measurement_result, is_blocked = self.take_measurement(self._selected_target_name)

        if is_blocked:
            self.update_blocklist_target_found(self._selected_target_name)

        if self.verbose and self.logfile:
            self.logfile.write(str(self._selected_target) + LOG_FILE_DELIMITER + str(measurement_result) + "\n")

        # call before observe
        is_optimal = self.is_optimal_action(selected_arm_seq)

        observed_value = self.observe(self._selected_arm_key, measurement_result)
        assert observed_value is not None, "Missing observed value, expecting a float"

        self.propagate_rewards(self._selected_arm_key)
        self.disable_target(self._selected_target)
        self.update_optimal_value()

        # set selected nodes explored
        for n in (selected_arm_seq + [self._selected_target]):
            self.action_space.get(n)[action_space_module.EXPLORED] = True

        return {"action": self._selected_arm_key,
                "target": self._selected_target_name,
                "reward": round(measurement_result, 2),
                "q_value": round(observed_value, 2),
                "is_blocked": 1 if is_blocked else 0,
                "is_optimal": 1 if is_optimal else 0,
                "coverage": self.get_blocklist_coverage()
                }

    def run(self, *args, **kwargs) -> pd.DataFrame:
        return self.model_runner.run(self, *args, **kwargs)


class ParserOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.params = {}

    def add_arguments(self):
        self.parser.add_argument("-tf", "--target_feature",
                                 choices=[TARGET_FEATURE__DOMAIN, TARGET_FEATURE__SERVICE_IP],
                                 type=str, default=TARGET_FEATURE__DOMAIN)

        # Hyperparameters
        self.parser.add_argument("-E", "--episodes", type=int, default=1)

        # Either run with max measurements or specify the number of measurements
        self.parser.add_argument("-M", "--maxmeasurements", action="store_true", default=False)
        self.parser.add_argument("-m", "--measurements", type=int, default=100)

        # Output and log files
        self.parser.add_argument("-o", "--outfile", type=str, default="")
        self.parser.add_argument("-v", "--verbose", action="store_true", default=False)
        self.parser.add_argument("-l", "--logfile", type=str, default="")

        # Ground truth file for evaluation
        self.parser.add_argument("-g", "--groundtruth", type=str)

        # action space
        self.parser.add_argument("-a", "--action_space_file", type=str)
        self.parser.add_argument("-asp", "--action_space_multi_parents", action="store_true", default=False)

        # features - at least one feature is required
        self.parser.add_argument("-f", "--features", nargs='+', default="categories")

        # How to treat unknowns - default as "Empty"
        self.parser.add_argument("-u", "--consider_unknown", default=UNKNOWN_EMPTY)

        # sample by rank
        self.parser.add_argument("-sr", "--sample_by_target_rank", action="store_true", default=False)

        # processes for episodes
        self.parser.add_argument("-np", "--num_of_processes_for_episodes", type=int, default=1)

    # Must be called after add_arguments to make sense.
    def set_params(self, args):
        self.params["target_feature"] = args.target_feature
        self.params["num_episodes"] = args.episodes

        measurements_per_episode = "run_until_exhaustion" if args.maxmeasurements else args.measurements
        self.params["measurements_per_episode"] = measurements_per_episode

        if args.outfile:
            dir_name = os.path.dirname(args.outfile)
            self.params["output_directory"] = dir_name
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        #stderr = sys.stderr
        #if args.logfile:
        #    stderr = open(args.logfile, "w")
        self.params["verbose"] = args.verbose
        self.params["outfile_csv"] = args.outfile + ".csv"
        #self.params["logfile"] = stderr
        self.params["ground_truth_path"] = args.groundtruth
        self.params["action_space_file"] = args.action_space_file
        self.params["features"] = args.features  # This is a list
        self.params["consider_unknown"] = args.consider_unknown
        self.params["sample_by_target_rank"] = args.sample_by_target_rank
        # self.params["input_dir_path"] = args.inputdirpath

        # action space
        self.params["action_space_multi_parents"] = args.action_space_multi_parents
        self.params["num_of_processes_for_episodes"] = args.num_of_processes_for_episodes

    def parse(self):
        self.add_arguments()
        args = self.parser.parse_args()
        self.set_params(args)

        return self.params


def run_episode(model: Model,
                episode_idx: int) -> list:

    episode_stats = []
    for model.current_epoch_num in range(model.measurements_per_episode):
        if model.can_step():
            episode_stat = {"episode": episode_idx,
                            "time": model.current_epoch_num + 1}
            episode_stat.update(model.step())
            episode_stats.append(episode_stat)
        else:
            break

        if model.current_epoch_num != 0 and model.current_epoch_num % 100 == 0:
            print(f"Episode Process {os.getpid()} - Done with {model.current_epoch_num} iterations")

    return episode_stats


class ModelRunBase:
    """
    This class implements the running logic
    """

    def set_measurements_per_episode(self, model: Model, num_data: int):
        if model.measurements_per_episode == "run_until_exhaustion" or model.measurements_per_episode > num_data:
            model.measurements_per_episode = num_data
        assert model.measurements_per_episode <= num_data

    def set_action_space(self, model: Model,
                         action_space_df: pd.DataFrame,
                         features: typing.List[str],
                         action_space_klass: typing.Callable):
        initial_value_estimate = action_space_module.DEFAULT_Q_VALUE
        if hasattr(model, "initial_value_estimate"):
            initial_value_estimate = getattr(model, "initial_value_estimate")

        action_value_file = None
        if hasattr(model, "action_value_file"):
            action_value_file = getattr(model, "action_value_file")

        # build the action space
        model.action_space = action_space_klass(model.output_directory,
                                                action_space_df,
                                                features,
                                                model.target_feature,
                                                default_q_value=initial_value_estimate,
                                                multiple_parents=model.action_space_multi_parents,
                                                action_value_file=action_value_file)

    def run_episodes(self,
                     model: Model,
                     save_stats: bool = False) -> pd.DataFrame:

        if len(model.blocklist_unique_count) == 0:
            model.set_blocklist_unique_counts_based_on_action_space()

        episode_all_stats = []
        for episode_idx in range(model.num_episodes):
            episode_stats = run_episode(model, episode_idx+1)
            episode_all_stats += episode_stats
            if episode_idx < model.num_episodes - 1:
                model.reset()

        stat_df = pd.DataFrame(columns=list(episode_all_stats[0].keys()))
        stat_df = stat_df.from_dict(episode_all_stats)

        if save_stats:
            print(f"Episode Process {os.getpid()} - Saving stats and model")
            model.save()
            stat_df.to_csv(model.outfile, index=False)

        return stat_df

    def run(self, model: Model,
            action_space_klass: typing.Callable = action_space_module.ActionSpaceBase,
            save_stats: bool = True) -> pd.DataFrame:

        num_data, action_space_df = run_preprocessor(model.action_space_file, model.features, model.consider_unknown)

        self.set_measurements_per_episode(model, num_data)
        self.set_action_space(model, action_space_df, model.features, action_space_klass)
        return self.run_episodes(model, save_stats=save_stats)


class DynModelRun(ModelRunBase):
    """
    This class does NOT limit the model.measurements_per_episode to num_data because are not removing data anymore when trying it,
    so the number of iterations can be > than num_data
    """

    def set_measurements_per_episode(self, model: Model, num_data: int):
        if model.measurements_per_episode == "run_until_exhaustion":
            model.measurements_per_episode = num_data


class DynOrderedModelRun(DynModelRun):
    """
    This class will run the data based on the number of unique dates in the dataset
    It will ignore given measurements and run_until_exhaustion
    """

    def set_measurements_per_episode(self, model: Model, num_data: int):
        assert "date" in model.blocklist.columns, f"Expected date column in {model.blocklist.head()}"
        assert hasattr(model, 'per_date_threshold'), "Expected model to have per_date_threshold attr"

        unique_dates = model.blocklist["date"].unique().tolist()
        model.measurements_per_episode = len(unique_dates) * model.per_date_threshold

        print(
            f"Episode Process {os.getpid()} - Found {len(unique_dates)} dates in ground truth and using {model.per_date_threshold} iterations per date")


def create_and_run_model(model_klass: typing.Callable,
                         params: dict,
                         addition_kwargs: dict = None,
                         addition_model_run_kwargs: dict = None) -> pd.DataFrame:
    #save_stats: bool = False

    if addition_kwargs:
        model = model_klass(params, **addition_kwargs)
    else:
        model = model_klass(params)

    if addition_model_run_kwargs:
        return model.run(**addition_model_run_kwargs)
    else:
        return model.run()


def run_multiprocessing(model_klass: typing.Callable,
                        params: dict,
                        addition_kwargs: dict = None,
                        addition_model_run_kwargs: dict = None):
    dfs = []
    episodes = params["num_episodes"]
    num_of_processes_for_episodes = params["num_of_processes_for_episodes"]
    # reset to 1, as each process will handle one episode only
    params["num_episodes"] = 1
    all_futures = dict()
    for episode_index in range(episodes):
        save_stats = episode_index + 1 == episodes
        if not addition_model_run_kwargs:
            addition_model_run_kwargs = dict()
        addition_model_run_kwargs["save_stats"] = save_stats
        all_futures[episode_index + 1] = create_and_run_model(
                                                         model_klass,
                                                         params,
                                                         addition_kwargs=addition_kwargs,
                                                         addition_model_run_kwargs=addition_model_run_kwargs)

    # aggregate results together
    for key, val in all_futures.items():
        df = val
        # set episode key to correct one
        df["episode"] = key
        dfs.append(df)
        print(f"Main process {os.getpid()} - Done with episode {key} out of {episodes}")
        sys.stdout.flush()

    df_merged = pd.concat(dfs)
    df_merged.to_csv(params["outfile_csv"], index=False)
