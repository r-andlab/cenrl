import logging
import random
import typing
from collections import deque

from baselines.baseline import Baseline, BaselineWithDate
from baselines.utils import COLUMN_NAME_DOMAIN, \
    multiple_episodes_baseline_by_klass
from common.utils import NO_DATE_BLOCKLIST


class TrancoByRankBaseline (Baseline):
    """
     (Tranco) Ranking: test an arm based on decreasing ranking
    """
    def run(self) -> list:
        results = []

        entry_count = dict()
        entries_in_order = self.df_action_space_file[COLUMN_NAME_DOMAIN].to_list()

        # use deque to quickly popleft
        entries_in_order = deque(entries_in_order)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(entries_in_order)

        for i in range(_measurements):

            # if nothing left to pick, then just keep current_value
            if not entries_in_order:
                results.append(self.take_action())
                continue

            entry = entries_in_order[0]
            if not self.with_replacement:
                entries_in_order.popleft()
            else:
                if entry not in entry_count:
                    entry_count[entry] = 0
                entry_count[entry] += 1

                # if we exceeded the number of times an entry can be used, then remove it
                if entry_count[entry] >= self.max_entry_retry and entry == entries_in_order[0]:
                    entries_in_order.popleft()

            if self.unique_dates:
                selected_date = random.choice(self.unique_dates)
                r = self.take_action(selected_target=entry, blocker_key=selected_date)
            else:
                r = self.take_action(selected_target=entry, blocker_key=NO_DATE_BLOCKLIST)

            results.append(r)

        return results


class TrancoByRandomRankBaseline (TrancoByRankBaseline):
    """
     (Tranco) Random selection: random select an arm to test
    """
    def run(self) -> list:
        results = []
        entry_count = dict()
        entries_in_order = self.df_action_space_file[COLUMN_NAME_DOMAIN].to_list()

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(entries_in_order)

        for i in range(_measurements):

            # if nothing left to pick, then just keep current_value
            if not entries_in_order:
                results.append(self.take_action())
                continue

            entry = random.choice(entries_in_order)
            if not self.with_replacement:
                entries_in_order.remove(entry)
            if entry not in entry_count:
                entry_count[entry] = 0
            entry_count[entry] += 1

            # if we exceeded the number of times an entry can be used, then remove it
            if entry_count[entry] >= self.max_entry_retry and self.with_replacement:
                entries_in_order.remove(entry)

            if self.unique_dates:
                selected_date = random.choice(self.unique_dates)
                r = self.take_action(selected_target=entry, blocker_key=selected_date)
            else:
                r = self.take_action(selected_target=entry, blocker_key=NO_DATE_BLOCKLIST)

            results.append(r)

        return results


class TrancoByRankDateBaseline (BaselineWithDate):
    """
    (Tranco) Date + Ranking: sequentially go through each date, test an arm based on decreasing ranking
    """

    def run(self) -> list:
        results = []

        entry_count = dict()
        entries_in_order_orig = self.df_action_space_file[COLUMN_NAME_DOMAIN].to_list()

        # use deque to quickly popleft
        entries_in_order = deque(entries_in_order_orig)

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(entries_in_order)

        unique_dates_tmp = deque(self.unique_dates)

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
            if not entries_in_order:
                results.append(self.take_action())
                continue

            entry = entries_in_order[0]
            if not self.with_replacement:
                entries_in_order.popleft()
            else:
                if entry not in entry_count:
                    entry_count[entry] = 0
                entry_count[entry] += 1

                # if we exceeded the number of times an entry can be used, then remove it
                if entry_count[entry] >= self.max_entry_retry and entry == entries_in_order[0]:
                    entries_in_order.popleft()

            r = self.take_action(selected_target=entry, blocker_key=self.current_date)
            results.append(r)

        return results


class TrancoByRandomRankDateBaseline (TrancoByRankDateBaseline):
    """
    (Tranco) Date + Random Ranking: sequentially go through each date, test an arm based on random ranking
    """
    def run(self) -> list:
        results = []

        entry_count = dict()
        entries_in_order = self.df_action_space_file[COLUMN_NAME_DOMAIN].to_list()

        _measurements = self.measurements
        if _measurements < 0:
            _measurements = len(entries_in_order)

        unique_dates_tmp = deque(self.unique_dates)

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
            if not entries_in_order:
                results.append(self.take_action())
                continue

            entry = random.choice(entries_in_order)
            if not self.with_replacement:
                entries_in_order.remove(entry)
            if entry not in entry_count:
                entry_count[entry] = 0
            entry_count[entry] += 1

            # if we exceeded the number of times an entry can be used, then remove it
            if entry_count[entry] >= self.max_entry_retry and self.with_replacement:
                entries_in_order.remove(entry)

            r = self.take_action(selected_target=entry, blocker_key=self.current_date)
            results.append(r)

        return results


def get_baseline_group_tranco_naive(action_space_file_path: str,
                                    episodes: int,
                                    measurements: int,
                                    logger: logging.Logger,
                                    adblockers: dict,
                                    max_entry_retry: int = 1,
                                    unique_dates: list = None,
                                    ) -> typing.List[typing.Tuple[str, list]]:
    baselines = []

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: tranco_by_rank_with_replacement")
        baseline_name = "tranco_by_rank_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRankBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)

        baselines.append((baseline_name, results))

    ############
    ############

    logger.info("calculating baseline: tranco_by_rank_no_replacement")
    baseline_name = "tranco_by_rank_no_replacement"
    results = multiple_episodes_baseline_by_klass(
        TrancoByRankBaseline,
        action_space_file_path,
        adblockers,
        episodes=episodes,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates,
        with_replacement=False)
    baselines.append((baseline_name, results))


    ############
    ############
    if max_entry_retry > 1:
        logger.info("calculating baseline: tranco_random_rank_with_replacement")
        # do it with replacement (basic random selection)
        baseline_name = "tranco_random_rank_with_replacement"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRandomRankBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            unique_dates=unique_dates)
        baselines.append((baseline_name, results))


    ############
    ############

    # do it again without replacement (basic random selection)
    # note that without replacement, the max measurements it will do is equal to the action_space_file_path list

    logger.info("calculating baseline: tranco_random_rank_no_replacement")
    baseline_name = "tranco_random_rank_no_replacement"

    results = multiple_episodes_baseline_by_klass(
        TrancoByRandomRankBaseline,
        action_space_file_path,
        adblockers,
        with_replacement=False,
        episodes=episodes,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        unique_dates=unique_dates
    )

    baselines.append((baseline_name, results))

    return baselines


def get_baseline_group_tranco_naive_date(action_space_file_path: str,
                                         episodes: int,
                                         measurements: int,
                                         logger: logging.Logger,
                                         adblockers: dict,
                                         unique_dates: list,
                                         max_entry_retry: int = 1,
                                         per_date_threshold: int = 2000) -> typing.List[
    typing.Tuple[str, list]]:
    baselines = []

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: tranco_by_rank_with_replacement_random_date")
        baseline_name = "tranco_by_rank_with_replacement_random_date"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRankDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=False,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

        logger.info("calculating baseline: tranco_by_rank_with_replacement_ordered_date")
        baseline_name = "tranco_by_rank_with_replacement_ordered_date"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRankDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=True,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

    logger.info("calculating baseline: tranco_by_rank_no_replacement_random_date")
    baseline_name = "tranco_by_rank_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        TrancoByRankDateBaseline,
        action_space_file_path,
        adblockers,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        with_replacement=False,
        use_ordered_dates=False,
        per_date_threshold=per_date_threshold,)
    baselines.append((baseline_name, results))

    logger.info("calculating baseline: tranco_by_rank_with_no_replacement_ordered_date")
    baseline_name = "tranco_by_rank_with_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        TrancoByRankDateBaseline,
        action_space_file_path,
        adblockers,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        with_replacement=False,
        use_ordered_dates=True,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    ############
    ############

    if max_entry_retry > 1:
        logger.info("calculating baseline: tranco_random_rank_with_replacement_random_date")
        baseline_name = "tranco_random_rank_with_replacement_random_date"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRandomRankDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=False,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

        logger.info("calculating baseline: tranco_random_rank_with_replacement_ordered_date")
        baseline_name = "tranco_random_rank_with_replacement_ordered_date"
        results = multiple_episodes_baseline_by_klass(
            TrancoByRandomRankDateBaseline,
            action_space_file_path,
            adblockers,
            episodes=episodes,
            unique_dates=unique_dates,
            measurements=measurements,
            max_entry_retry=max_entry_retry,
            use_ordered_dates=True,
            per_date_threshold=per_date_threshold)
        baselines.append((baseline_name, results))

    logger.info("calculating baseline: tranco_random_rank_no_replacement_random_date")
    baseline_name = "tranco_random_rank_no_replacement_random_date"
    results = multiple_episodes_baseline_by_klass(
        TrancoByRandomRankDateBaseline,
        action_space_file_path,
        adblockers,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        with_replacement=False,
        use_ordered_dates=False,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    logger.info("calculating baseline: tranco_random_rank_no_replacement_ordered_date")
    baseline_name = "tranco_random_rank_no_replacement_ordered_date"
    results = multiple_episodes_baseline_by_klass(
        TrancoByRandomRankDateBaseline,
        action_space_file_path,
        adblockers,
        episodes=episodes,
        unique_dates=unique_dates,
        measurements=measurements,
        max_entry_retry=max_entry_retry,
        with_replacement=False,
        use_ordered_dates=True,
        per_date_threshold=per_date_threshold)
    baselines.append((baseline_name, results))

    return baselines
