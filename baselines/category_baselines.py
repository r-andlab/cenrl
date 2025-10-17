import logging
import random
import typing
from collections import deque

from baselines.baseline import Baseline, BaselineWithDate
from baselines.utils import get_unique_categories_to_avg_ranking, \
    COLUMN_NAME_CATEGORIES, COLUMN_NAME_DOMAIN, multiple_episodes_baseline_by_klass
from common.utils import NO_DATE_BLOCKLIST


class CategoriesBaseline (Baseline):
    """
     Select a category, then select a target
     When use_avg_ranking = False:
        - Randomly choose a category and choose ALL domains from the category and test, then move to the next random category
     When use_avg_ranking = True:
        - Choose the category with the highest average ranking and choose ALL domains from the category, then move to the category with the next highest average ranking.

    """

    def __init__(self, *args, use_avg_ranking: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_avg_ranking = use_avg_ranking

    def run(self) -> list:
        results = []

        entry_count = dict()
        unique_categories = get_unique_categories_to_avg_ranking(self.df_action_space_file)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(self.df_action_space_file)

        _curr_selected_category = None
        category_rank = None
        for i in range(_measurements):
            # if nothing left to pick, then just keep current_value
            if len(self.df_action_space_file) == 0 or not unique_categories:
                results.append(self.take_action())
                continue

            entry = None
            while not entry:
                if not unique_categories:
                    break

                if not _curr_selected_category:
                    if not self.use_avg_ranking:
                        _curr_selected_category, category_rank = random.choice(unique_categories)
                    else:
                        # take at top of list
                        _curr_selected_category, category_rank = unique_categories[0]
                    # print(f"_curr_selected_category {_curr_selected_category} ")

                avail_entries = self.df_action_space_file[
                    self.df_action_space_file[COLUMN_NAME_CATEGORIES].str.contains(_curr_selected_category)]
                if len(avail_entries) > 0:
                    entry = avail_entries.sample(1)[COLUMN_NAME_DOMAIN].iloc[0]
                    if not self.with_replacement:
                        self.df_action_space_file = self.df_action_space_file[self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                    else:
                        if entry not in entry_count:
                            entry_count[entry] = 0
                        entry_count[entry] += 1

                        # if we exceeded the number of times an entry can be used, then remove it
                        if entry_count[entry] >= self.max_entry_retry:
                            self.df_action_space_file = self.df_action_space_file[
                                self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                else:
                    unique_categories.remove((_curr_selected_category, category_rank))
                    _curr_selected_category = None

            if self.unique_dates:
                selected_date = random.choice(self.unique_dates)
                r = self.take_action(selected_target=entry, blocker_key=selected_date)
            else:
                r = self.take_action(selected_target=entry, blocker_key=NO_DATE_BLOCKLIST)

            results.append(r)

        return results


class CategoriesRoundRobinBaseline (CategoriesBaseline):
    """
     Select a category round-robin
    When use_avg_ranking=False:
        - Randomly choose a category and choose 1 domain from the category, then move to the next random category
    When use_avg_ranking=True:
        - Choose the category with the highest average ranking and choose 1 domain from the category, then move to the
        category with the next highest average ranking.
    """
    def run(self) -> list:

        results = []

        entry_count = dict()
        unique_categories = get_unique_categories_to_avg_ranking(self.df_action_space_file)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(self.df_action_space_file)

        cat_index = 0
        for i in range(_measurements):

            # if nothing left to pick, then just keep current_value
            if len(self.df_action_space_file) == 0 or not unique_categories:
                results.append(self.take_action())
                continue

            entry = None

            while not entry:
                if not unique_categories:
                    break

                if not self.use_avg_ranking:
                    selected_category, category_rank = random.choice(unique_categories)
                else:
                    # round-robin the highest category
                    selected_category, category_rank = unique_categories[cat_index]

                avail_entries = self.df_action_space_file[
                    self.df_action_space_file[COLUMN_NAME_CATEGORIES].str.contains(selected_category)]
                if len(avail_entries) > 0:
                    entry = avail_entries.sample(1)[COLUMN_NAME_DOMAIN].iloc[0]
                    if not self.with_replacement:
                        self.df_action_space_file = self.df_action_space_file[self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                    else:
                        if entry not in entry_count:
                            entry_count[entry] = 0
                        entry_count[entry] += 1

                        # if we exceeded the number of times an entry can be used, then remove it
                        if entry_count[entry] >= self.max_entry_retry:
                            self.df_action_space_file = self.df_action_space_file[
                                self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]

                    cat_index += 1
                else:
                    unique_categories.remove((selected_category, category_rank))
                    # if we remove a category, keep cat_index same

                # reset cat_index
                if cat_index >= len(unique_categories):
                    cat_index = 0

            if self.unique_dates:
                selected_date = random.choice(self.unique_dates)
                r = self.take_action(selected_target=entry, blocker_key=selected_date)
            else:
                r = self.take_action(selected_target=entry, blocker_key=NO_DATE_BLOCKLIST)

            results.append(r)

        return results


class CategoriesDateBaseline (BaselineWithDate):
    """
    (Tranco) Date + Categories: sequentially go through each date, pick a category and then target
    When use_avg_ranking = False:
        - Randomly choose a category and choose ALL domains from the category and test, then move to the next random category
    When use_avg_ranking = True:
        - Choose the category with the highest average ranking and choose ALL domains from the category, then move to the category with the next highest average ranking.
    """
    def __init__(self, *args, use_avg_ranking: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_avg_ranking = use_avg_ranking

    def run(self) -> list:
        results = []

        entry_count = dict()
        unique_categories = get_unique_categories_to_avg_ranking(self.df_action_space_file)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(self.df_action_space_file)

        unique_dates_tmp = deque(self.unique_dates)

        _curr_selected_category = None
        category_rank = None
        for i in range(_measurements):

            if i % self.per_date_threshold == 0:
                entry_count.clear()

                if not unique_dates_tmp:
                    break

                if self.use_ordered_dates and unique_dates_tmp:
                    self.current_date = unique_dates_tmp.popleft()
                    # print(f"{i} new date {selected_date}")

            if not self.use_ordered_dates:
                self.current_date = random.choice(unique_dates_tmp)

            # if nothing left to pick, then just keep current_value
            if len(self.df_action_space_file) == 0 or not unique_categories:
                results.append(self.take_action())
                continue

            entry = None
            while not entry:
                if not unique_categories:
                    break
                if not _curr_selected_category:
                    if not self.use_avg_ranking:
                        _curr_selected_category, category_rank = random.choice(unique_categories)
                    else:
                        # take at top of list
                        _curr_selected_category, category_rank = unique_categories[0]
                    # print(f"_curr_selected_category {_curr_selected_category} ")

                avail_entries = self.df_action_space_file[
                    self.df_action_space_file[COLUMN_NAME_CATEGORIES].str.contains(_curr_selected_category)]
                if len(avail_entries) > 0:
                    entry = avail_entries.sample(1)[COLUMN_NAME_DOMAIN].iloc[0]
                    if not self.with_replacement:
                        self.df_action_space_file = self.df_action_space_file[self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                    else:
                        if entry not in entry_count:
                            entry_count[entry] = 0
                        entry_count[entry] += 1

                        # if we exceeded the number of times an entry can be used, then remove it
                        if entry_count[entry] >= self.max_entry_retry:
                            self.df_action_space_file = self.df_action_space_file[
                                self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                else:
                    unique_categories.remove((_curr_selected_category, category_rank))
                    _curr_selected_category = None

            r = self.take_action(selected_target=entry, blocker_key=self.current_date)
            results.append(r)

        return results


class CategoriesRoundRobinDateBaseline (CategoriesDateBaseline):
    """
    (Tranco) Date + Categories: sequentially go through each date, pick a category round-robin and then target
    When use_avg_ranking=False:
        - Randomly choose a category and choose 1 domain from the category, then move to the next random category
    When use_avg_ranking=True:
        - Choose the category with the highest average ranking and choose 1 domain from the category, then move to the
        category with the next highest average ranking.
    """

    def run(self) -> list:
        results = []

        entry_count = dict()
        unique_categories = get_unique_categories_to_avg_ranking(self.df_action_space_file)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(self.df_action_space_file)

        unique_dates_tmp = deque(self.unique_dates)

        cat_index = 0
        selected_date = None
        for i in range(_measurements):

            if i % self.per_date_threshold == 0:
                entry_count.clear()

                if not unique_dates_tmp:
                    break

                if self.use_ordered_dates and unique_dates_tmp:
                    self.current_date = unique_dates_tmp.popleft()
                    # print(f"{i} new date {selected_date}")

            if not self.use_ordered_dates:
                self.current_date = random.choice(unique_dates_tmp)

            # if nothing left to pick, then just keep current_value
            if len(self.df_action_space_file) == 0 or not unique_categories:
                results.append(self.take_action())
                continue

            entry = None

            while not entry:
                if not unique_categories:
                    break

                if not self.use_avg_ranking:
                    selected_category, category_rank = random.choice(unique_categories)
                else:
                    # round-robin the highest category
                    selected_category, category_rank = unique_categories[cat_index]

                avail_entries = self.df_action_space_file[
                    self.df_action_space_file[COLUMN_NAME_CATEGORIES].str.contains(selected_category)]
                if len(avail_entries) > 0:
                    entry = avail_entries.sample(1)[COLUMN_NAME_DOMAIN].iloc[0]
                    if not self.with_replacement:
                        self.df_action_space_file = self.df_action_space_file[self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]
                    else:
                        if entry not in entry_count:
                            entry_count[entry] = 0
                        entry_count[entry] += 1

                        # if we exceeded the number of times an entry can be used, then remove it
                        if entry_count[entry] >= self.max_entry_retry:
                            self.df_action_space_file = self.df_action_space_file[
                                self.df_action_space_file[COLUMN_NAME_DOMAIN] != entry]

                    cat_index += 1
                else:
                    unique_categories.remove((selected_category, category_rank))
                    # if we remove a category, keep cat_index same

                # reset cat_index
                if cat_index >= len(unique_categories):
                    cat_index = 0

            r = self.take_action(selected_target=entry, blocker_key=self.current_date)
            results.append(r)

        return results


def get_baseline_group_categories(action_space_file_path: str,
                                  episodes: int, measurements: int,
                                  logger: logging.Logger,
                                  adblockers: dict,
                                  max_entry_retry: int = 1,
                                  unique_dates: list = None) -> typing.List[typing.Tuple[str, list]]:
    baselines = []

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_with_replacement")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            CategoriesBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_no_replacement")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_no_replacement"
    results = multiple_episodes_baseline_by_klass(
        CategoriesBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates)
    baselines.append((baseline_name, results))

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_avg_ranking_with_replacement")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            CategoriesBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_avg_ranking_no_replacement")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_no_replacement"
    results = multiple_episodes_baseline_by_klass(
        CategoriesBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates)
    baselines.append((baseline_name, results))

    return baselines


def get_baseline_group_round_robin_categories(action_space_file_path: str,
                                              episodes: int, measurements: int,
                                              logger: logging.Logger,
                                              adblockers: dict,
                                              max_entry_retry: int = 1,
                                              unique_dates: list = None) -> typing.List[
    typing.Tuple[str, list]]:
    baselines = []

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_round_robin_with_replacement")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_round_robin_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_round_robin_no_replacement")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_round_robin_no_replacement"

    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates)
    baselines.append((baseline_name, results))

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_avg_ranking_round_robin_with_replacement")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_round_robin_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_avg_ranking_round_robin_no_replacement")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_round_robin_no_replacement"
    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates)
    baselines.append((baseline_name, results))

    return baselines


def get_baseline_group_categories_date(action_space_file_path: str,
                                       episodes: int, measurements: int,
                                       logger: logging.Logger,
                                       adblockers: dict,
                                       unique_dates: list,
                                       max_entry_retry: int = 1,
                                       per_date_threshold: int = 2000) -> typing.List[
    typing.Tuple[str, list]]:
    baselines = []

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_with_replacement_random_date")
        baseline_name = "categories_with_replacement_random_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            per_date_threshold=per_date_threshold,
            use_ordered_dates=False)
        baselines.append((baseline_name, results))

        logger.info("calculating baseline: categories_with_replacement_ordered_date")
        baseline_name = "categories_with_replacement_ordered_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            per_date_threshold=per_date_threshold,
            use_ordered_dates=True)
        baselines.append((baseline_name, results))


    ############
    ############

    logger.info("calculating baseline: categories_no_replacement_random_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        per_date_threshold=per_date_threshold,
        use_ordered_dates=False
    )
    baselines.append((baseline_name, results))


    logger.info("calculating baseline: categories_no_replacement_ordered_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        per_date_threshold=per_date_threshold,
        use_ordered_dates=True
    )
    baselines.append((baseline_name, results))


    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_avg_ranking_with_replacement_random_date")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_with_replacement_random_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            per_date_threshold=per_date_threshold,
            use_ordered_dates=False
        )
        baselines.append((baseline_name, results))

        logger.info("calculating baseline: categories_avg_ranking_with_replacement_ordered_date")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_with_replacement_ordered_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            per_date_threshold=per_date_threshold,
            use_ordered_dates=True
        )
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_avg_ranking_no_replacement_random_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        per_date_threshold=per_date_threshold,
        use_ordered_dates=False)
    baselines.append((baseline_name, results))


    logger.info("calculating baseline: categories_avg_ranking_no_replacement_ordered_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        unique_dates=unique_dates,
        episodes=episodes,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        per_date_threshold=per_date_threshold,
        use_ordered_dates=True)
    baselines.append((baseline_name, results))

    return baselines


def get_baseline_group_round_robin_categories_date(action_space_file_path: str,
                                                   episodes: int, measurements: int,
                                                   logger: logging.Logger,
                                                   adblockers: dict,
                                                   unique_dates: list,
                                                   max_entry_retry: int = 1,
                                                   per_date_threshold: int = 2000
                                                   ) -> typing.List[
    typing.Tuple[str, list]]:
    baselines = []

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_round_robin_with_replacement_random_dates")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_round_robin_with_replacement_random_dates"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=False,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

        logger.info("calculating baseline: categories_round_robin_with_replacement_ordered_dates")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_round_robin_with_replacement_ordered_dates"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=True,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_round_robin_no_replacement_random_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_round_robin_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        use_ordered_dates=False,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    logger.info("calculating baseline: categories_round_robin_no_replacement_ordered_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_round_robin_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        use_ordered_dates=True,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: categories_avg_ranking_round_robin_with_replacement_random_date")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_round_robin_with_replacement_random_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=False,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))


        logger.info("calculating baseline: categories_avg_ranking_round_robin_with_replacement_ordered_date")
        # do it with replacement (basic categories random selection)
        baseline_name = "categories_avg_ranking_round_robin_with_replacement_ordered_date"
        results = multiple_episodes_baseline_by_klass(
            CategoriesRoundRobinDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            use_avg_ranking=True,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=True,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: categories_avg_ranking_round_robin_no_replacement_random_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_round_robin_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        use_ordered_dates=False,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    logger.info("calculating baseline: categories_avg_ranking_round_robin_no_replacement_ordered_date")
    # do it with replacement (basic categories random selection)
    baseline_name = "categories_avg_ranking_round_robin_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        CategoriesRoundRobinDateBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        use_avg_ranking=True,
        max_entry_retry=max_entry_retry,
        use_ordered_dates=True,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    return baselines
