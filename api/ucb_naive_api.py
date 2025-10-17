import typing

from models.base.model import run_multiprocessing
from models.ucb.ucb_naive import UCBNaive, UCBNaiveParserOptions


# Example of how to integrate CenRL UCBNaive into am active censorship measurement platform for real-world measurements
class UCBNaiveAPI(UCBNaive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Step 1: Add any necessary initializations
        # e.g., hyperquack.start() or ooni.start()

    def take_measurement(self, selected_target) -> typing.Tuple[float, bool]:
        # Step 2: call you platform's measurement API directly and use the selected_target.
        # e.g., hyperquack.start_scan(selected_target) or ooni.scan(selected_target)
        # The selected_target is what has been chosen by CenRL to be measured for censorship (e.g., a website). 
        # returns a Tuple (reward (0=not blocked, 1=blocked), bool for if blocked)
        raise NotImplementedError()

    # The below methods are necessary for controlled environments, ignore them.
    def parse_block_list(self):
        pass
    
    def get_blocklist_coverage(self) -> float:
        return 0

    def set_blocklist_unique_counts_based_on_action_space(self):
        pass

    def init_blockers(self):
        pass

    def update_blocklist_target_found(self, target_found: str):
        pass
    

if __name__ == "__main__":
    parser = UCBNaiveParserOptions()
    params = parser.parse()
    run_multiprocessing(UCBNaiveAPI, params)

