# Logging configuration
from logging import config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "rich_fmt": {"format": "[%(name)s][%(funcName)s]: %(message)s"},
    },
    "handlers": {
        "rich": {
            "level": "DEBUG",
            "formatter": "rich_fmt",
            "omit_repeated_times": False,
            "show_path": False,
            "class": "rich.logging.RichHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["rich"],
            "level": "DEBUG",
        },
    },
}
config.dictConfig(LOGGING_CONFIG)
del config, LOGGING_CONFIG
