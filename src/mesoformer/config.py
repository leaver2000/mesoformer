import os

from .typing import DictStrAny
from .utils import find, load_toml

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "metadata.toml")


def get_config() -> DictStrAny:
    return load_toml(FILE)


def get_datasets() -> list[DictStrAny]:
    return get_config()["datasets"]


def get_dataset(title: str) -> DictStrAny:
    return find(lambda x: x["title"] == title.upper(), get_datasets())
