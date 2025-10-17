import concurrent.futures
import typing
from random import randrange

import numpy as np

from models.base.action_space import NODE_TYPE_KEY, POSSIBLE_TARGET_FEATURES, ACTION_ATTEMPTS, Q_VALUE, SLEEPING
from models.base.model import Model, ParserOptions, run_multiprocessing

np.seterr(divide="ignore", invalid="ignore")


class UCBNaive(Model):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.c = params["c"]
        self.step_size = params["step_size"]
        self.initial_value_estimate = params["initial_value_estimate"]

        # used to calculate exploration, which may go
        # out of sync with self.current_epoch_num later
        self.exploration_epoch_num = self.current_epoch_num

    def reset(self):
        super().reset()
        self.exploration_epoch_num = self.current_epoch_num

    def choose_arm(self) -> typing.List[str]:

        def _choose_arm_by_layer(_node_keys: typing.List[str],
                                 _value_estimates: np.array,
                                 _action_attempts: np.array) -> str:
            exploration = np.log(self.exploration_epoch_num + 1) / _action_attempts
            # if arm a has been selected 0 times, then selecting it is considered a
            # maximizing action (Sutton and Barto 36). So we use np.inf here.
            exploration[np.isnan(exploration)] = np.inf
            exploration = np.sqrt(exploration) * self.c
            # print(f"exploration: {exploration}")
            # print(f"value estimates: {_value_estimates}")
            # print(f"Done iteration {self.exploration_epoch_num + 1}")
            # print(f"action attempts {_action_attempts}")
            bound = _value_estimates + exploration
            #print(f"Done bound {bound}")
            argmax_indices = np.argwhere(bound == np.max(bound))
            selected_index = argmax_indices[randrange(0, len(argmax_indices))][0]
            return _node_keys[selected_index]

        # loop through the hierarchy and choose largest Q + UCB (stop when we reach target nodes)
        source = self.action_space.get_root()
        selected_arm = None
        reached_target_nodes = False
        selected_arms_history = []
        while not reached_target_nodes:
            immediate_children = []
            action_attempts = []
            value_estimates = []

            for succ in self.action_space.get_graph().successors(source):
                succ_data = self.action_space.get(succ)
                if succ_data[NODE_TYPE_KEY] in POSSIBLE_TARGET_FEATURES:
                    reached_target_nodes = True
                    break

                immediate_children.append(succ)
                value_estimates.append(succ_data[Q_VALUE])
                action_attempts.append(succ_data[ACTION_ATTEMPTS])

            if reached_target_nodes:
                break

            if immediate_children and not reached_target_nodes:
                selected_arm = _choose_arm_by_layer(immediate_children,
                                                    np.array(value_estimates),
                                                    np.array(action_attempts))
                selected_arms_history.append(selected_arm)

            source = selected_arm

        #print(f"{self.current_epoch_num}: final chosen arm {selected_arm}")

        if self.verbose and self.logfile:
            self.logfile.write("Max|" + str(selected_arm) + "|")

        # NOTE: must update the ACTION_ATTEMPTS for all arms except for last (which will be updated in the observe)
        for a in selected_arms_history[:-1]:
            self.action_space.get(a)[ACTION_ATTEMPTS] += 1

        self.last_selected_arm_index = selected_arms_history[-1]
        return selected_arms_history

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        n_data = self.action_space.get(selected_arm)
        n_data[ACTION_ATTEMPTS] += 1

        g = self.step_size
        if g is None or g == 0:
            g = 1 / n_data[ACTION_ATTEMPTS]
        q = n_data[Q_VALUE]
        q += g * (measurement_result - q)
        n_data[Q_VALUE] = round(q, 2)

        return n_data[Q_VALUE]

    def step(self) -> dict:
        # we gotta update exploration_epoch_num ourselves
        self.exploration_epoch_num += 1
        return super().step()


class UCBNaiveParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument("-c", "--c", type=float, default=1)
        self.parser.add_argument("-s", "--stepsize", type=float, default=None)
        self.parser.add_argument("-V", "--initialvalueestimate", type=float, default=1)
        self.parser.add_argument("-X", "--actionvaluefile", type=str, default=None)

    def set_params(self, args):
        super().set_params(args)
        self.params["c"] = args.c
        self.params["step_size"] = args.stepsize
        self.params["initial_value_estimate"] = args.initialvalueestimate
        self.params["action_value_file"] = args.actionvaluefile


if __name__ == "__main__":
    parser = UCBNaiveParserOptions()
    params = parser.parse()
    run_multiprocessing(UCBNaive, params)
