import datetime
import os
import typing

import numpy as np
import pandas as pd

COLUMN_NAME_DOMAIN = "domain"
COLUMN_NAME_CATEGORIES = "categories"
COLUMN_NAME_ENTITY = "entity"
COLUMN_NAME_RANK = "rank"
COLUMN_NAME_DATE = "date"



def get_unique_categories_to_avg_ranking(df: pd.DataFrame) -> typing.List[typing.Tuple[str, int]]:
    categories_to_ranking = dict()
    df_categories = df[COLUMN_NAME_CATEGORIES].apply(eval)
    for _cat_list in df_categories:
        for c in _cat_list:
            if c not in categories_to_ranking:
                # drop dups based on domain just in case
                match_rows = df[df[COLUMN_NAME_CATEGORIES].str.contains(c)].drop_duplicates(subset=[COLUMN_NAME_DOMAIN])
                c_rank = int(np.average(match_rows[COLUMN_NAME_RANK].to_list()))
                categories_to_ranking[c] = c_rank

    # flatten, sort by v, ascending because lower number means better ranking
    cat_to_rank_list = [(k, v) for k, v in categories_to_ranking.items()]
    cat_to_rank_list.sort(key=lambda x: x[1])

    # print(cat_to_rank_list)

    return cat_to_rank_list


def get_unique_entities_to_avg_ranking(df: pd.DataFrame) \
        -> typing.List[typing.Tuple[str, int]]:
    entities_to_ranking = dict()
    for entity in df[COLUMN_NAME_ENTITY].unique().tolist():
        # drop dups based on domain just in case
        if pd.notna(entity):
            match_rows = df[df[COLUMN_NAME_ENTITY] == entity].drop_duplicates(subset=[COLUMN_NAME_DOMAIN])
            rank = int(np.average(match_rows[COLUMN_NAME_RANK].to_list()))
            entities_to_ranking[entity] = rank

    # flatten, sort by v, ascending because lower number means better ranking
    entity_to_rank_list = [(k, v) for k, v in entities_to_ranking.items()]
    entity_to_rank_list.sort(key=lambda x: x[1])

    # print(entity_to_rank_list)

    return entity_to_rank_list


def get_unique_dates_from_csv_file_path(csv_file_path: str,
                                        delimiter: str = "|",
                                        date_column_name: str = "date",
                                        date_format: str = "%Y%m%d") -> typing.Tuple[pd.DataFrame, list]:
    df = pd.read_csv(csv_file_path, delimiter=delimiter)
    unique_dates = []
    if date_column_name in df.columns.tolist():
        df[date_column_name] = df[date_column_name].apply(lambda x: datetime.datetime.strptime(str(x), date_format))
        unique_dates = df[date_column_name].unique().tolist()
        unique_dates.sort()

    return df, unique_dates


def save_baseline_raw(data: list, output_directory: str, output_name: str) -> typing.Tuple[pd.DataFrame, str]:
    file_name = output_directory + os.sep + output_name + ".csv"
    df = pd.DataFrame(columns=list(data[0].keys()))
    df = df.from_dict(data)
    df.to_csv(output_directory + os.sep + output_name + ".csv", index=False)
    return df, file_name


def multiple_episodes_baseline_by_klass(baseline_klass: typing.Callable,
                                        *args,
                                        episodes: int = 20,
                                        **kwargs) -> list:
    result_lists = []

    for x in range(episodes):
        b = baseline_klass(*args, **kwargs, episode=x)
        r = b.run()
        result_lists += r

    return result_lists


def multiple_episodes_baseline(baseline_func: typing.Callable,
                               *args,
                               episodes: int = 20,
                               np_aggregrate_func: typing.Callable = np.mean,
                               **kwargs) -> list:
    result_lists = []

    for x in range(episodes):
        r = baseline_func(*args, **kwargs)
        result_lists.append(np.array(r))

    aggregated_results = [np_aggregrate_func(x) for x in zip(*result_lists)]

    return aggregated_results


def save_baseline(baseline_tuple, output_directory: str) -> str:
    baseline_name, x_values, y_values, aoc = baseline_tuple

    # remove the aoc part of the label
    file_name = baseline_name.split("(")[0].strip()
    output_file_path = output_directory + os.sep + file_name + ".csv"

    # build csv
    rows = []
    for x, y in zip(x_values, y_values):
        rows.append({"x": x, "y": y})

    df = pd.DataFrame(columns=list(rows[0].keys()))
    df = df.from_dict(rows)
    df.to_csv(output_file_path, sep="|", index=False)

    return output_file_path


def save_baselines_to_one_file(baseline_tuples, output_directory: str) -> str:
    rows = []
    for baseline_tuple in baseline_tuples:
        baseline_name, x_values, y_values, aoc = baseline_tuple

        # build csv
        for x, y in zip(x_values, y_values):
            rows.append({"x": x, "y": y, "name": baseline_name})

    output_file_path = output_directory + os.sep + "baselines.csv"

    df = pd.DataFrame(columns=list(rows[0].keys()))
    df = df.from_dict(rows)
    df.to_csv(output_file_path, sep="|", index=False)

    return output_file_path


def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return round(area, 2)
