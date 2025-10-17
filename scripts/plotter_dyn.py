# !/usr/bin/python
import argparse
import glob
import logging
import os
import typing
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from baselines.utils import integrate
from common.utils import get_cumulative_avg_across_episodes, get_avg_across_episodes, TARGET_FEATURE__DOMAIN, \
    FEATURE_CATEGORIES, UNKNOWN_EMPTY
from models.base.preprocessor import process_feature
from plot_utils import plot_blocked_occurrences, AOC_INDEX, plot_avg_coverage, plot_avg_reward


def create_new_df_for_feature_counting(results_file_path: str,
                                       feature: str,
                                       action_space_file_path: str,
                                       consider_blocked_only: bool = True,
                                       consider_unknown: str = UNKNOWN_EMPTY,
                                       target_feature: str = TARGET_FEATURE__DOMAIN) -> typing.Optional[pd.DataFrame]:
    df_results = pd.read_csv(results_file_path, index_col=False)
    df_action_space = pd.read_csv(action_space_file_path, delimiter="|", index_col=False)

    # try different delimiters
    if target_feature not in df_action_space.columns:
        df_action_space = pd.read_csv(action_space_file_path, delimiter=",", index_col=False)

    assert target_feature in df_action_space.columns, f"Target feature {target_feature} not in action space dataframe"

    if feature in df_action_space.columns:
        df_action_space = process_feature(df_action_space, feature, consider_unknown)

    if consider_blocked_only:
        df_results = df_results[df_results["is_blocked"] == 1]

    new_rows = []
    for row_dict in df_results.to_dict(orient='records'):
        match_feature_value = df_action_space[df_action_space[target_feature] == row_dict["target"]][feature]
        if not isinstance(match_feature_value, list):
            match_feature_value = [match_feature_value]
        for v in match_feature_value:
            new_row = dict()
            new_row.update(row_dict)
            new_row[feature] = v
            new_rows.append(new_row)

    df_new = None
    if new_rows:
        df_new = pd.DataFrame(columns=list(new_rows[0].keys()))
        df_new = df_new.from_dict(new_rows)
    return df_new


def plot_before_after_aoc_threshold(result_files_found: typing.List[str],
                                    results_directory: str,
                                    output_file_name: str,
                                    aoc_threshold: int,
                                    action_space_file_path: str,
                                    logger: logging.Logger,
                                    feature: str = FEATURE_CATEGORIES,
                                    target_feature: str = TARGET_FEATURE__DOMAIN):
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams['pdf.fonttype'] = 42

    for file_path in result_files_found:
        df_results = create_new_df_for_feature_counting(file_path,
                                                        feature,
                                                        action_space_file_path,
                                                        target_feature=target_feature)
        if df_results is None:
            continue

        df_results.loc[df_results["time"] <= aoc_threshold, "time_threshold"] = f"before {aoc_threshold}"
        df_results.loc[df_results["time"] > aoc_threshold, "time_threshold"] = f"after {aoc_threshold}"
        df_results_block = df_results[(df_results["is_blocked"] == 1)]
        #action_counts = df_results_block["action"].value_counts()
        #action_order = action_counts.index

        time_threshold_to_sort = f"before {aoc_threshold}"
        action_counts_sorted = df_results_block[df_results_block["time_threshold"] == time_threshold_to_sort][
            "action"].value_counts().index.tolist()
        action_counts_sorted_other = df_results_block[df_results_block["time_threshold"] != time_threshold_to_sort][
            "action"].value_counts().index.tolist()

        missing_action_counts = set(action_counts_sorted_other) - set(action_counts_sorted)
        logger.info(f"Missing action count indices {missing_action_counts}, adding back in")
        action_counts_sorted += [x for x in action_counts_sorted_other if x in missing_action_counts]

        plt.figure(figsize=(5, 19))
        line_name = os.path.basename(file_path)
        line_name = str(line_name).split("_params_")[-1]
        # print(df_results_block.head())
        g = sns.countplot(data=df_results_block,
                          y="action",
                          hue="time_threshold",
                          hue_order=[f"before {aoc_threshold}", f"after {aoc_threshold}"],
                          order=action_counts_sorted)
        plt.xlabel("Count")
        plt.ylabel("Action")
        plt.xscale('log')
        plt.title(line_name)
        sns.despine()
        g.xaxis.grid(alpha=0.3)
        g.yaxis.grid(alpha=0.3)
        plot_output_directory = results_directory + os.sep + "plots"
        os.makedirs(plot_output_directory, exist_ok=True)
        output_file_name = os.path.basename(file_path).replace(".csv", ".pdf")
        feature_clean = feature.replace(" ", "_")
        g.figure.savefig(plot_output_directory + os.sep + "action_count_" + feature_clean + "_" + output_file_name,
                         bbox_inches="tight")

        plt.clf()


def plot_is_optimal(result_files_found: typing.List[str],
                    results_directory: str,
                    output_file_name: str,
                    aoc_threshold: int,
                    logger: logging.Logger):
    results = []
    for file_path in result_files_found:
        df_results = pd.read_csv(file_path, index_col=False)
        y_results = get_avg_across_episodes(df_results, "is_optimal")
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:aoc_threshold], y_results[:aoc_threshold])
        results.append((file_path, x_values, y_results, aoc))

    # sort results by area under curve
    results = sorted(results, key=lambda x: x[AOC_INDEX], reverse=True)

    max_colors = len(results)
    color_map = matplotlib.colormaps["jet"]

    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(20, 20))

    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.',
          '-', '--', ':', '-.']
    linecycler = cycle(ls)
    g = None
    i = 0

    logger.info("plotting optimal results")
    for file_path, x, y, aoc in results:
        _color = color_map(((5 * i) % max_colors) / max_colors)
        line_name = os.path.basename(file_path)
        line_name = str(line_name).split("_params_")[-1]
        line_name += f" (auc={aoc:,})"

        g = sns.lineplot(x=x, y=y,
                         label=line_name, linestyle=next(linecycler),
                         alpha=0.7, linewidth=0.5, color=_color)
        i += 1

    g.set(ylim=(0, 1))
    plt.xlabel("Time")
    plt.ylabel("Optimal Action %")
    plt.title(output_file_name)
    sns.despine()
    g.xaxis.grid(alpha=0.3)
    g.yaxis.grid(alpha=0.3)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plot_output_directory = results_directory + os.sep + "plots"
    os.makedirs(plot_output_directory, exist_ok=True)
    g.figure.savefig(plot_output_directory + os.sep + "optimal_" + output_file_name,
                     bbox_inches="tight")

    # IMPORTANT clear of main plot
    plt.clf()

    logger.info(f"plotting individual result per plot for {len(results)} results")

    # plot all individually
    linecycler = cycle(ls)
    g = None
    i = 0
    for file_path, x, y, aoc in results:
        _color = color_map(((5 * i) % max_colors) / max_colors)
        line_name = os.path.basename(file_path)
        line_name = str(line_name).split("_params_")[-1]
        line_name += f" (auc={aoc:,})"
        g = sns.lineplot(x=x, y=y,
                         label=line_name, linestyle=next(linecycler),
                         alpha=1, linewidth=3, color=_color)
        i += 1
        plt.xlabel("Time")
        plt.ylabel("Optimal Action %")
        plt.title(line_name)
        sns.despine()
        g.set(ylim=(0, 1))
        g.xaxis.grid(alpha=0.3)
        g.yaxis.grid(alpha=0.3)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plot_output_directory = results_directory + os.sep + "plots"
        os.makedirs(plot_output_directory, exist_ok=True)
        output_file_name = os.path.basename(file_path).replace(".csv", ".pdf")
        g.figure.savefig(plot_output_directory + os.sep + "optimal_" + output_file_name,
                         bbox_inches="tight")

        plt.clf()

    logger.info(f"Files outputted to {plot_output_directory}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot data of RL results and baselines')

    # REQUIRED
    parser.add_argument('--output_file_name',
                        required=True,
                        help='output name of the main results')
    parser.add_argument('--target_feature',
                        required=False,
                        default=TARGET_FEATURE__DOMAIN,
                        help='target feature name')
    parser.add_argument('--features',
                        required=False,
                        type=str,
                        help='List of features, separated by space')
    parser.add_argument('--results_directory',
                        required=True,
                        help='directory of results to plot')
    parser.add_argument('--results_prefix',
                        required=True,
                        help='suffix to result files to find')
    parser.add_argument('--ground_truth_file_path',
                        required=False,
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
    parser.add_argument('--include_baselines',
                        action='store_true',
                        help='whether we should do the baselines as well')
    parser.add_argument('--selected_target_max_try',
                        required=False,
                        default=10,
                        type=int,
                        help='number of time a selected target is used (for baselines only)')
    parser.add_argument('--selected_target_max_try_ordered',
                        required=False,
                        default=1,
                        type=int,
                        help='number of time a selected target is used (for baselines with ordered date only)')
    parser.add_argument('--log_level',
                        default="INFO",
                        help='Log level')

    args = parser.parse_args()
    print(args)

    output_file_name = args.output_file_name
    results_directory = args.results_directory
    results_prefix = args.results_prefix
    measurements = args.measurements
    episodes = args.episodes
    action_space_file_path = args.action_space_file_path
    per_date_threshold = args.per_date_threshold
    selected_target_max_try = args.selected_target_max_try
    target_feature = args.target_feature
    features = args.features.split() if args.features else []

    assert os.path.exists(results_directory), f"Results directory not found: {results_directory}"

    # set up logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.root.handlers = []
    logging.basicConfig(
        handlers=[logging.FileHandler(results_directory + os.sep + "plotting.log", mode="w"), logging.StreamHandler()],
        format='%(asctime)s %(module)s - %(message)s', level=numeric_level)

    logger = logging.getLogger(__name__)

    result_files_found = sorted(glob.glob(results_directory + os.sep + f"{results_prefix}*.csv"))
    if features:
        logger.info(f"features passed in {features}")

    for f in features:
        plot_before_after_aoc_threshold(result_files_found,
                                        results_directory,
                                        output_file_name,
                                        args.aoc_threshold,
                                        action_space_file_path,
                                        logger,
                                        feature=f,
                                        target_feature=target_feature)

    plot_is_optimal(result_files_found,
                    results_directory,
                    output_file_name,
                    args.aoc_threshold,
                    logger)

    results = []
    for file_path in result_files_found:
        df_results = pd.read_csv(file_path, index_col=False)
        y_results = get_avg_across_episodes(df_results, "reward")
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])
        results.append((file_path, x_values, y_results, aoc))

    # sort results by area under curve
    results = sorted(results, key=lambda x: x[AOC_INDEX], reverse=True)

    plot_avg_reward(results,
                    results_directory,
                    output_file_name,
                    logger)

    results = []
    for file_path in result_files_found:
        df_results = pd.read_csv(file_path, index_col=False)
        y_results = get_avg_across_episodes(df_results, "coverage") * 100
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])
        results.append((file_path, x_values, y_results, aoc))

    # sort results by area under curve
    results = sorted(results, key=lambda x: x[AOC_INDEX], reverse=True)

    plot_avg_coverage(results,
                      results_directory,
                      output_file_name,
                      logger)

    results = []
    for file_path in result_files_found:
        df_results = pd.read_csv(file_path, index_col=False)
        y_results = get_cumulative_avg_across_episodes(df_results,
                                                       property_column_name="is_blocked")
        x_values = list(range(len(y_results)))
        aoc = integrate(x_values[:args.aoc_threshold], y_results[:args.aoc_threshold])
        results.append((file_path, x_values, y_results, aoc))

    # sort results by area under curve
    results = sorted(results, key=lambda x: x[AOC_INDEX], reverse=True)
    plot_blocked_occurrences(
        results,
        results_directory,
        output_file_name,
        logger
    )


if __name__ == "__main__":
    main()
