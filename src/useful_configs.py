import config

# TODO: These config names are crap. Maybe change them?

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
        "fr_fo_ratio_gt_50",
        "fr_fo_ratio"
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
        "fr_fo_ratio"
        # "listed_count" Dropping this cause wtf does it mean?
    ],
    classify_on="is_bot"
)

PRUNE = config.ConfigVars(
    name="pruned",
    cols_to_build=[
        "is_bot",
        "has_default_profile_image",
        "geo_enabled",
        "has_name",
    ],
    cols_to_keep=[
        "is_bot",
        "has_default_profile_image",
        "geo_enabled",
        "has_name",
    ],
    classify_on="is_bot"
)

MINIMAL_PREEXIST = config.ConfigVars(
    name="only friends/followers/statuses",
    cols_to_build=[
        "is_bot",
    ],
    cols_to_keep=[
        "is_bot",
        "followers_count",
        "friends_count",
        "statuses_count",
    ],
    classify_on="is_bot"
)
