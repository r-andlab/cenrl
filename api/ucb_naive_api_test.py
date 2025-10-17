import typing

from api.ucb_naive_api import UCBNaiveAPI
from models.base.model import run_multiprocessing
from models.ucb.ucb_naive import UCBNaiveParserOptions


# Example of how to integrate CenRL UCBNaive into am active censorship measurement platform for real-world measurements
class UCBNaiveAPITest(UCBNaiveAPI):

    # since this is a test, we simply randomize the results
    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        # Step 2: call you platform's measurement API directly and use the selected_target.
        # e.g., hyperquack.start_scan(selected_target) or ooni.scan(selected_target)
        # The selected_target is what has been chosen by CenRL to be measured for censorship (e.g., a website). 
        # returns a Tuple (reward (0=not blocked, 1=blocked), bool for if blocked)
        import random
        reward = random.choice([0, 1])  # randomly pick 0 or 1
        return reward, reward == 1


if __name__ == "__main__":
    parser = UCBNaiveParserOptions()
    params = parser.parse()
    run_multiprocessing(UCBNaiveAPITest, params)

