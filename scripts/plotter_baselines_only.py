# !/usr/bin/python
import argparse
import logging
import os
import typing

import pandas as pd

from baselines.category_baselines import get_baseline_group_categories, get_baseline_group_round_robin_categories
from baselines.entity_baselines import get_baseline_group_entities, get_baseline_group_round_robin_entities
from baselines.tranco_naive_baselines import get_baseline_group_tranco_naive
from baselines.utils import get_unique_dates_from_csv_file_path, \
    COLUMN_NAME_CATEGORIES, integrate, save_baseline_raw, COLUMN_NAME_ENTITY
from common.utils import NO_DATE_BLOCKLIST, TARGET_FEATURE__DOMAIN, get_cumulative_avg_across_episodes, \
    get_avg_across_episodes
from models.base.utils_adblocker import create_adblocker_with_ground_truth
from plot_utils import plot_blocked_occurrences, plot_avg_coverage, plot_avg_reward, AOC_INDEX


def get_baselines_data_no_dates(ground_truth_file_path: str, action_space_file_path: str,
                                episodes: int, measurements: int, aoc_threshold: int,
                                logger: logging.Logger,
                                output_directory: str,
                                max_entry_retry: int = 1,
                                target_feature: str = TARGET_FEATURE__DOMAIN) -> typing.List[
    typing.Tuple[str, list]]:
    df_action_space_file = pd.read_csv(action_space_file_path, delimiter="|", index_col=False)
    df_blocklist, _ = get_unique_dates_from_csv_file_path(ground_truth_file_path)
    adblockers = {NO_DATE_BLOCKLIST: create_adblocker_with_ground_truth(df_blocklist, target_feature)}

    baselines = []

    # NOTE: no dates are passed this is not based on dates
    baselines += get_baseline_group_tranco_naive(action_space_file_path,
                                                 episodes,
                                                 measurements,
                                                 logger,
                                                 adblockers,
                                                 max_entry_retry=max_entry_retry)


    if COLUMN_NAME_CATEGORIES in df_action_space_file.columns:

        baselines += get_baseline_group_categories(action_space_file_path,
                                                   episodes,
                                                   measurements,
                                                   logger,
                                                   adblockers,
                                                   max_entry_retry=max_entry_retry)

        baselines += get_baseline_group_round_robin_categories(action_space_file_path,
                                                               episodes,
                                                               measurements,
                                                               logger,
                                                               adblockers,
                                                               max_entry_retry=max_entry_retry)

    else:
        logger.warning(f"No {COLUMN_NAME_CATEGORIES} categories found in {action_space_file_path}")


    if COLUMN_NAME_ENTITY in df_action_space_file.columns:

        baselines += get_baseline_group_entities(
            action_space_file_path,
            episodes,
            measurements,
            logger,
            adblockers,
            max_entry_retry=max_entry_retry)

        baselines += get_baseline_group_round_robin_entities(action_space_file_path,
                                                             episodes,
                                                             measurements,
                                                             logger,
                                                             adblockers,
                                                             max_entry_retry=max_entry_retry)

    else:
        logger.warning(f"No {COLUMN_NAME_ENTITY} categories found in {action_space_file_path}")

    ############
    ############

    return baselines


def main():
    parser = argparse.ArgumentParser(
        description='Plot baselines')

    # REQUIRED
    parser.add_argument('--output_file_name',
                        required=True,
                        help='output name of the main results')
    parser.add_argument('--output_directory',
                        required=True,
                        help='output directory')
    parser.add_argument('--ground_truth_file_path',
                        required=True,
                        help='path to ground truth file')
    parser.add_argument('--action_space_file_path',
                        required=True,
                        help='path to file used in building the action space of the RL experiments')
    parser.add_argument('--measurements',
                        required=True,
                        type=int,
                        help='number of measurements used in results and used for baseline running')
    parser.add_argument('--episodes',
                        required=True,
                        type=int,
                        help='number of episodes used in results and used for baseline running')
    parser.add_argument('--per_date_threshold',
                        required=False,
                        default=-1,
                        type=int,
                        help='number of time steps per unique date if ground truth has date column, '
                             'otherwise will be ignored')
    parser.add_argument('--aoc_threshold',
                        required=False,
                        default=4000,
                        type=int,
                        help='number of time steps to calculate area under curve')
    parser.add_argument('--selected_target_max_try',
                        required=False,
                        default=10,
                        type=int,
                        help='number of time a selected target is used (for baselines only)')
    parser.add_argument('--log_level',
                        default="INFO",
                        help='Log level')

    args = parser.parse_args()
    print(args)

    output_file_name = args.output_file_name
    ground_truth_file_path = args.ground_truth_file_path
    measurements = args.measurements
    episodes = args.episodes
    action_space_file_path = args.action_space_file_path
    per_date_threshold = args.per_date_threshold
    selected_target_max_try = args.selected_target_max_try

    assert os.path.exists(ground_truth_file_path), f"Ground truth file not found: {ground_truth_file_path}"

    # set up logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.root.handlers = []
    logging.basicConfig(
        handlers=[logging.FileHandler(args.output_directory + os.sep + "baselinesonly.log", mode="w"),
                  logging.StreamHandler()],
        format='%(asctime)s %(module)s - %(message)s', level=numeric_level)

    logger = logging.getLogger(__name__)

    # still need this to match up with other baselines using dates
    # for Baselines before, we may need to override measurements
    if per_date_threshold > 0:
        df, unique_dates = get_unique_dates_from_csv_file_path(ground_truth_file_path)
        if unique_dates:
            logger.warning(
                f"Overriding {measurements} with {per_date_threshold * len(unique_dates)}, due to unique dates")
            measurements = per_date_threshold * len(unique_dates)

    baseline_output_directory = args.output_directory + os.sep + "baselines"
    os.makedirs(baseline_output_directory, exist_ok=True)
    baselines = get_baselines_data_no_dates(ground_truth_file_path,
                                            action_space_file_path,
                                            episodes,
                                            measurements,
                                            args.aoc_threshold,
                                            logger,
                                            baseline_output_directory,
                                            max_entry_retry=selected_target_max_try)

    baselines_aggregates = []
    # save
    for baseline_name, baseline_result in baselines:
        df_baseline, file_name = save_baseline_raw(baseline_result, baseline_output_directory, baseline_name)

        aggregated_results = dict()

        # is_blocked aggregated results
        y_results = get_cumulative_avg_across_episodes(df_baseline,
                                                       property_column_name="is_blocked")
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])

        aggregated_results["is_blocked"] = (file_name, x_values, y_results, aoc)

        # average reward aggregated results
        y_results = get_avg_across_episodes(df_baseline, property_column_name="reward")
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])
        aggregated_results["avg_reward"] = (file_name, x_values, y_results, aoc)

        # coverage aggregated results
        y_results = get_avg_across_episodes(df_baseline, property_column_name="coverage") * 100
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])
        aggregated_results["coverage"] = (file_name, x_values, y_results, aoc)
        baselines_aggregates.append((baseline_name, aggregated_results))

    blocked_ocurrences_results = [r.get("is_blocked") for _, r in baselines_aggregates]
    blocked_ocurrences_results = sorted(blocked_ocurrences_results, key=lambda x: x[AOC_INDEX], reverse=True)

    plot_blocked_occurrences(blocked_ocurrences_results,
                             args.output_directory,
                             output_file_name,
                             logger)

    blocked_coverage_results = [r.get("coverage") for _, r in baselines_aggregates]
    blocked_coverage_results = sorted(blocked_coverage_results, key=lambda x: x[AOC_INDEX], reverse=True)
    plot_avg_coverage(blocked_coverage_results,
                      args.output_directory,
                      output_file_name,
                      logger)

    blocked_reward_results = [r.get("avg_reward") for _, r in baselines_aggregates]
    blocked_reward_results = sorted(blocked_reward_results, key=lambda x: x[AOC_INDEX], reverse=True)

    plot_avg_reward(blocked_reward_results,
                    args.output_directory,
                    output_file_name,
                    logger)


if __name__ == "__main__":
    main()
