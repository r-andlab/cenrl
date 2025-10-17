import collections
import typing

import numpy as np
from random import randrange

from models.base.action_space import POSSIBLE_TARGET_FEATURES, Q_VALUE, ACTION_ATTEMPTS, NODE_TYPE_KEY
from models.base.model import Model, ParserOptions, run_multiprocessing


class EpsilonGreedySampling(Model):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.epsilon = params["epsilon"]
        self.step_size = params["step_size"]
        self.initial_value_estimate = params["initial_value_estimate"]

    def choose_arm(self) -> typing.List[str]:

        def _choose_arm_by_layer(_node_keys: typing.List[str],
                                 _value_estimates: np.array) -> str:

            argmax_indices = np.argwhere(_value_estimates == np.max(_value_estimates))
            selected_index = argmax_indices[randrange(0, len(argmax_indices))][0]
            return _node_keys[selected_index]

        source = self.action_space.get_root()
        selected_arm = None
        selected_arms_history = []
        d = collections.deque()
        d.append(source)
        if np.random.uniform() < self.epsilon:
            while d:
                n = d.popleft()
                # Perform true random sampling on categories.
                if self.action_space.get_graph().out_degree(n) > 0:
                    selected_arm_tmp = self.action_space.sample_successors(n)[0]
                    if self.action_space.get_graph().out_degree(selected_arm_tmp) > 0:
                        selected_arm = selected_arm_tmp
                        d.append(selected_arm)
                        selected_arms_history.append(selected_arm)
        else:
            # loop through the hierarchy and choose largest Q (stop when we reach target nodes)
            reached_target_nodes = False
            while not reached_target_nodes:
                immediate_children = []
                value_estimates = []

                for succ in self.action_space.get_graph().successors(source):
                    succ_data = self.action_space.get(succ)
                    if succ_data[NODE_TYPE_KEY] in POSSIBLE_TARGET_FEATURES:
                        reached_target_nodes = True
                        break

                    immediate_children.append(succ)
                    value_estimates.append(succ_data[Q_VALUE])

                if reached_target_nodes:
                    break

                if immediate_children and not reached_target_nodes:
                    selected_arm = _choose_arm_by_layer(immediate_children,
                                                        np.array(value_estimates))
                    selected_arms_history.append(selected_arm)

                source = selected_arm
                # print(f"{self.current_epoch_num}: chose arm {selected_arm}")

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
        n_data[Q_VALUE] = q
        return n_data[Q_VALUE]


class EpsilonGreedySamplingParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument("-e", "--epsilon", type=float, default=0.2)
        self.parser.add_argument("-s", "--stepsize", type=float, default=None)
        self.parser.add_argument("-V", "--initialvalueestimate", type=float, default=1)
        self.parser.add_argument("-X", "--actionvaluefile", type=str, default=None)

    def set_params(self, args):
        super().set_params(args)
        self.params["epsilon"] = args.epsilon
        self.params["step_size"] = args.stepsize
        self.params["initial_value_estimate"] = args.initialvalueestimate
        self.params["action_value_file"] = args.actionvaluefile


if __name__ == "__main__":
    parser = EpsilonGreedySamplingParserOptions()
    params = parser.parse()

    run_multiprocessing(EpsilonGreedySampling, params)

