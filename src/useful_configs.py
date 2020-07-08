


# None of the features we're keeping
# actually split the data that well.
SIMPLE_CONFIG = config.ConfigVars(
    cols_to_build=[
        "is_bot",
        "has_default_profile_image", 
        "no_screen_name",
        "language_not_empty",
        "description_contains_url",
        "description_length",
        "geo_enabled"
    ],
    cols_to_keep=[
        "is_bot",
        "has_default_profile_image", 
        "no_screen_name",
        "language_not_empty",
        "description_contains_url",
        "description_length",
    ],
    classify_on="is_bot"
)

