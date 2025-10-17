import logging
import os
import re
from itertools import cycle

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

AOC_INDEX = 3

def get_episode_num(line):
    return re.search("([0-9])+", line).group()


def get_measurement_result(result_file):
    REWARD_INDEX_BACK = -2
    aggregated_results = []
    with open(result_file) as f:
        episode_results = []
        episode_idx = 1
        # need to properly aggregate episodes
        for line in f:
            if int(get_episode_num(line)) != episode_idx:
                episode_idx = int(get_episode_num(line))
                if not aggregated_results:
                    aggregated_results = episode_results
                else:
                    aggregated_results = [measurement[0] + measurement[1] for measurement in
                                          zip(aggregated_results, episode_results)]
                episode_results = []
            # [-1] is \n, so we want [-2]
            line_split = line.split("|")
            episode_results.append(float(line_split[REWARD_INDEX_BACK]))

    if not aggregated_results:
        aggregated_results = episode_results
    else:
        aggregated_results = [measurement[0] + measurement[1] for measurement in
                              zip(aggregated_results, episode_results)]

    aggregated_results = np.array(aggregated_results).astype('float64')
    aggregated_results /= float(episode_idx)
    aggregated_results = np.cumsum(aggregated_results)
    return aggregated_results


def plot_blocked_occurrences(results: list,
                             results_directory: str,
                             output_file_name: str,
                             logger: logging.Logger):
    """
    Expect results to be list of tuple: (file_path, x_values, y_results, aoc))
    """

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

    logger.info("plotting main results")
    for file_path, x, y, aoc in results:
        _color = color_map(((5 * i) % max_colors) / max_colors)
        line_name = os.path.basename(file_path)
        line_name = str(line_name).split("_params_")[-1]
        line_name += f" (auc={aoc:,})"
        g = sns.lineplot(x=x, y=y,
                         label=line_name, linestyle=next(linecycler),
                         alpha=0.7, linewidth=0.5, color=_color)
        i += 1

    plt.xlabel("Time")
    plt.ylabel("Occurrences of Censorship Found")
    plt.title(output_file_name)
    sns.despine()
    g.xaxis.grid(alpha=0.3)
    g.yaxis.grid(alpha=0.3)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plot_output_directory = results_directory + os.sep + "plots"
    os.makedirs(plot_output_directory, exist_ok=True)
    g.figure.savefig(plot_output_directory + os.sep + output_file_name,
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
        plt.ylabel("Occurrences of Censorship Found")
        plt.title(line_name)
        sns.despine()
        g.xaxis.grid(alpha=0.3)
        g.yaxis.grid(alpha=0.3)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plot_output_directory = results_directory + os.sep + "plots"
        os.makedirs(plot_output_directory, exist_ok=True)
        output_file_name = os.path.basename(file_path).replace(".csv", ".pdf")
        g.figure.savefig(plot_output_directory + os.sep + output_file_name,
                         bbox_inches="tight")

        plt.clf()

    logger.info(f"Files outputted to {plot_output_directory}")


def plot_avg_coverage(results: list,
                      results_directory: str,
                      output_file_name: str,
                      logger: logging.Logger):

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

    logger.info("plotting coverage results")
    for file_path, x, y, aoc in results:
        _color = color_map(((5 * i) % max_colors) / max_colors)
        line_name = os.path.basename(file_path)
        line_name = str(line_name).split("_params_")[-1]
        line_name += f" (auc={aoc:,})"

        g = sns.lineplot(x=x, y=y,
                         label=line_name, linestyle=next(linecycler),
                         alpha=0.7, linewidth=0.5, color=_color)
        i += 1

    g.set(ylim=(0, 100))
    plt.xlabel("Time")
    plt.ylabel("Coverage %")
    plt.title(output_file_name)
    sns.despine()
    g.xaxis.grid(alpha=0.3)
    g.yaxis.grid(alpha=0.3)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plot_output_directory = results_directory + os.sep + "plots"
    os.makedirs(plot_output_directory, exist_ok=True)
    g.figure.savefig(plot_output_directory + os.sep + "coverage_" + output_file_name,
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
        plt.ylabel("Coverage %")
        plt.title(line_name)
        sns.despine()
        g.set(ylim=(0, 100))
        g.xaxis.grid(alpha=0.3)
        g.yaxis.grid(alpha=0.3)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plot_output_directory = results_directory + os.sep + "plots"
        os.makedirs(plot_output_directory, exist_ok=True)
        output_file_name = os.path.basename(file_path).replace(".csv", ".pdf")
        g.figure.savefig(plot_output_directory + os.sep + "coverage_" + output_file_name,
                         bbox_inches="tight")

        plt.clf()

    logger.info(f"Files outputted to {plot_output_directory}")


def plot_avg_reward(results: list,
                    results_directory: str,
                    output_file_name: str,
                    logger: logging.Logger):

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

    logger.info("plotting reward results")
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
    plt.ylabel("Avg. Reward")
    plt.title(output_file_name)
    sns.despine()
    g.xaxis.grid(alpha=0.3)
    g.yaxis.grid(alpha=0.3)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plot_output_directory = results_directory + os.sep + "plots"
    os.makedirs(plot_output_directory, exist_ok=True)
    g.figure.savefig(plot_output_directory + os.sep + "reward_" + output_file_name,
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
        plt.ylabel("Avg. Reward")
        plt.title(line_name)
        sns.despine()
        g.set(ylim=(0, 1))
        g.xaxis.grid(alpha=0.3)
        g.yaxis.grid(alpha=0.3)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plot_output_directory = results_directory + os.sep + "plots"
        os.makedirs(plot_output_directory, exist_ok=True)
        output_file_name = os.path.basename(file_path).replace(".csv", ".pdf")
        g.figure.savefig(plot_output_directory + os.sep + "reward_" + output_file_name,
                         bbox_inches="tight")

        plt.clf()

    logger.info(f"Files outputted to {plot_output_directory}")
