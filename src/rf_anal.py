import data
import util
import config

from sklearn.ensemble import RandomForestClassifier


base_config = config.ConfigVars(
    to_sample={
        # Let's try out these two datasets.
        data.DataSets.GENUINE: 1000,
        data.DataSets.T_1: 1000
    },
    cols_to_build=data.TO_BUILD,
    cols_to_keep=data.TO_KEEP,
    classify_on="is_bot"
)


if __name__ == "__main__":
    pass
