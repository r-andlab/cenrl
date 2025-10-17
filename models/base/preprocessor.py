import argparse
import os.path
from ast import literal_eval

import pandas as pd

from common.utils import TARGET_FEATURE__DOMAIN, TARGET_FEATURE__SERVICE_IP, FEATURE_CATEGORIES, FEATURE_VPS, \
    UNKNOWN_EMPTY


def process_empty_category(category):
    if len(category) == 0:
        return ['Newly Seen Domains']
    return category


def process_unknown_feature(feature):
    if feature == '':
        return UNKNOWN_EMPTY
    return feature


def process_feature(df, feature, consider_unknown):
    match feature:
        case "categories":
            # needed because the df reads the entries in "categories" column in as string instead of list
            df[FEATURE_CATEGORIES] = df[FEATURE_CATEGORIES].apply(literal_eval)
            if consider_unknown == UNKNOWN_EMPTY:
                # Set unknown values to Newly Seen Domains
                df[FEATURE_CATEGORIES] = df[FEATURE_CATEGORIES].apply(process_empty_category)
            else:
                # Get rid of empty values
                df = df[df[FEATURE_CATEGORIES].str.len() != 0]
            df = df.explode(FEATURE_CATEGORIES, ignore_index=True)

        case _:
            if consider_unknown == UNKNOWN_EMPTY:
                df[feature] = df[feature].fillna('').apply(process_unknown_feature)
            else:
                # Get rid of empty values
                df = df[(df[feature].fillna('') != "")]
    return df


# NOTE: num_outcomes is legacy from when we were considering multiple outcome values
# (besides blocked / not blocked) for a measurement. We are not using this anymore, but
# keeping it here in case it is desirable later.
def preprocess(file, features, consider_unknown):

    file_name = os.path.basename(file)

    if f"{FEATURE_CATEGORIES}" in file_name:
        target_feature = TARGET_FEATURE__DOMAIN
    elif f"{FEATURE_VPS}" in file_name:
        target_feature = TARGET_FEATURE__SERVICE_IP
    else:
        raise Exception(f"Unrecognized input file {file}. Stopping run.\n"
                        "Preprocessing is currently implemented for category and vantage point anaylsis only,\n"
                        "and filenames need to include categories or vps respectively.")

    df = pd.read_csv(file, index_col=False, delimiter='|')
    for feature in features:
        df = process_feature(df, feature, consider_unknown)

    return df[target_feature].nunique(), df


def run_preprocessor(action_space_file, features, consider_unknown):
    return preprocess(action_space_file, features, consider_unknown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--action_space_file", type=str,
                        default="../../inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv")
    parser.add_argument("-f", "--features", nargs='+', default=["categories"])
    parser.add_argument("-u", "--consider_unknown", default=UNKNOWN_EMPTY)

    args = parser.parse_args()

    num_data, df = run_preprocessor(args.action_space_file, args.features, args.consider_unknown,
                                              )
    print(num_data)
