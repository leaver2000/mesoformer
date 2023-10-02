"""
all of the child dataset modules should implement the follow functions and classes.

## Dataset module template

### core.py

```
from __future__ import annotations


import pandas as pd

from .constants import CHANNELS, CATALOG_SCHEMA, 


def read_catalog(dataset: str) -> pd.DataFrame:
    ...


def download_dataset(dataset: str, file: str) -> None:
    ...


def download_catalog(catalog: pd.DataFrame, dataset: str) -> None:
    ...


class CatalogInterface:...    
```


## constants.py

```
from __future__ import annotations


from ...generic import ChannelEnum


class Channel(ChannelEnum):
    A = "A"
    B = "B"
    C = "C"

    
CHANNELS = A, B, C = tuple(Channel)
```

"""
