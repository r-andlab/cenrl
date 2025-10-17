import unittest
import pandas as pd
from common.utils import reward_in_blocklist


class TestRewardInBlocklist(unittest.TestCase):
    def test_data_frame_with_domain(self):
        df = pd.DataFrame({"domain": ["*.0132.fun", "abc.google.com"]})
        self.assertEqual(reward_in_blocklist(df, "www.0132.fun", "domain"), 1)

    def test_data_frame_without_domain_1(self):
        df = pd.DataFrame({"domain": ["*.0132.fun", "abc.google.com"]})
        self.assertEqual(reward_in_blocklist(df, "0132.fun", "domain"), 0)

    def test_data_frame_without_domain_2(self):
        df = pd.DataFrame({"domain": ["*.0132.fun", "abc.google.com"]})
        self.assertEqual(reward_in_blocklist(df, "xyz.google.com", "domain"), 0)

    def test_data_frame_without_domain_3(self):
        df = pd.DataFrame({"domain": ["*.0132.fun", "abc.google.com"]})
        self.assertEqual(reward_in_blocklist(df, "google.com", "domain"), 0)

    def test_data_frame_with_domain_2(self):
        df = pd.DataFrame({"domain": ["*.0132.fun", "abc.google.com"]})
        self.assertEqual(reward_in_blocklist(df, "xyz.abc.google.com", "domain"), 1)

    def test_list_without_domain(self):
        lst = ["*.google.com", "*.yahoo.com"]
        self.assertEqual(reward_in_blocklist(lst, "www.google.com", "domain"), 1)

if __name__ == "__main__":
    unittest.main()
