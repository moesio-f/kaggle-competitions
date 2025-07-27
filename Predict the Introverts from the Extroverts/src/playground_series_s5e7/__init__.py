# Logging configuration
from logging import config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {"format": "[%(name)s][%(funcName)s]: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "omit_repeated_times": False,
            "show_path": False,
            "class": "rich.logging.RichHandler",
        },
    },
    "loggers": {
        "playground_series_s5e7": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
config.dictConfig(LOGGING_CONFIG)
del config, LOGGING_CONFIG
