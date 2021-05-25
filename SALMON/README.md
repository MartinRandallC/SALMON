# SALMON

```
dataset/   # Datasets
lib/       # Python code common to notebooks, scripts, ...
notebooks/ # Notebooks (data exploration, ...)
```

## Prerequisites

#### pipenv

The Python 3 dependencies are managed using [pipenv](https://pipenv.readthedocs.io/en/latest/) which can be installed by running:

```bash
pip install pipenv
```

## Notebooks

Subsequents commands should be run in the repository root:

```bash
git clone git@github.com:MartinRandallC/SALMON.git
cd SALMON
```

On the first run:

```bash
pipenv sync
pipenv run jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

To run jupyter lab:

```bash
pipenv run jupyter lab
```

## Policies

To update the receding horizon policy, place the new files in `lib/policies/receding_horizon/` and make the following changes:

```python
# RecedingHorizon.py
from . import MarkovChainM as mc   # import MarkovChainM as mc
from .NetworkEnvironmentM import * # from NetworkEnvironmentM import *

# NetworkEnvironmentM.py
from . import MarkovChainM as mc   # import MarkovChainM as mc
```
