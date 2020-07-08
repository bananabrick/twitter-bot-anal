import config

# Uses everything we've built and
# everything I think we should keep
# from the raw data.
ALL = config.ConfigVars(
    name="all_features",
    cols_to_build=[
        "is_bot",
        "has_default_profile_image", 
        "no_screen_name",
        "language_not_empty",
        "description_contains_url",
        "description_length",
        "geo_enabled",
        "has_name",
        "fr_fo_ratio_gt_100",
        "fr_fo_ratio_gt_50"
    ],
    cols_to_keep=[
        "is_bot",
        "has_default_profile_image", 
        "no_screen_name",
        "language_not_empty",
        "description_contains_url",
        "description_length",
        "fr_fo_ratio_gt_100",
        "fr_fo_ratio_gt_50",
        "has_name",
        "geo_enabled",
        "followers_count",
        "friends_count",
        "statuses_count",
        "listed_count"
    ],
    classify_on="is_bot"
)

