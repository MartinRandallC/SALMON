import json
import time

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

from .models import HMM, HMMFilter


# MDP state


class DiscreteFilter:
    def __init__(self, model, z_init=0, τ_init=10000):
        self.delays = [state.mean for state in model.states]
        self.model = model
        self.z = z_init
        self.τ = τ_init

    def update(self, z):
        self.z = z
        self.τ = 0

    def predict(self):
        self.τ += 1

    @property
    def belief(self):
        return np.linalg.matrix_power(self.model.transmat, self.τ)[self.z, :]

    @property
    def expected_delay(self):
        return np.dot(self.belief, self.delays)


# Scenarios


class Scenario:
    def __init__(self, timeseries, models, name="Untitled"):
        assert len(timeseries) == len(models)
        self.timeseries = timeseries
        self.models = models
        self.name = name

    def __repr__(self):
        Ks = [len(model.states) for model in self.models]
        avg_delays = [np.round(np.mean(ts), 2) for ts in self.timeseries]
        return "{} scenario with {} paths\n- K = {}\n- Avg. delays = {}".format(
            self.name, self.n_paths, Ks, avg_delays
        )

    @property
    def n_paths(self):
        return len(self.timeseries)

    @classmethod
    def from_path(cls, path, dirty_check=False):
        path = Path(path)
        timeseries, models = [], []
        
        # Hack to make sure that the direct path is first
        # Cleanup that... :-)
        ts_files = list(path.glob("*.csv"))
        ts_files.sort(key=lambda x: len(x.name.split('_')))
        if dirty_check:
            print(ts_files[0].with_suffix('').name, path.name)
            assert(ts_files[0].with_suffix('').name == path.name)
    
        for ts_file in ts_files:
            timeseries.append(read_timeseries(ts_file, fillna=True))
            models.append(read_model(ts_file.with_suffix(".json")))
        return cls(timeseries, models, name=path.name)


# Simulator


class Logbook:
    
    def __init__(self, costs):
        self.costs = costs
        self.history = []
        
    def record(self, data):
        self.history.append(data)

    def get(self, key):
        return np.array([d.get(key) for d in self.history])
        
    def avg_cost(self):
        actions = self.get('action')
        return np.mean([np.dot(action, self.costs) for action in actions])
        
    def avg_delay(self):
        return np.mean(self.get('delay'))
    
    def avg_processing_time(self):
        return np.mean(self.get('processing_time'))
    
    def avg_n_measures(self):
        actions = self.get('action')
        return np.mean([np.sum(action) for action in actions])


class Simulator:
    
    def __init__(self, timeseries, models, costs):
        assert len(timeseries) == len(models) == len(costs)
        assert len(set(map(len, timeseries))) == 1

        self.timeseries = timeseries
        self.models = models
        self.costs = costs

    def benchmark(self, monitoring_policy, routing_policy):
        T = len(self.timeseries[0])

        # Filters
        # TODO: Start in stationnary dist. (τ_init = τ_max) ?
        hmm_filters = [HMMFilter(model) for model in self.models]
        mdp_filters = [DiscreteFilter(model, τ_init=0) for model in self.models]
        
        # History
        logbook = Logbook(costs=self.costs)
    
        for t in tqdm(range(T), leave=False):
            start_time = time.process_time_ns()
            action = monitoring_policy.action(mdp_filters)
            
            processing_time = time.process_time_ns() - start_time

            for i, a in enumerate(action):
                if a:
                    hmm_filters[i].update(self.timeseries[i][t])
                    mdp_filters[i].update(np.argmax(hmm_filters[i].belief))
                else:
                    hmm_filters[i].predict()
                    mdp_filters[i].predict()

            expected_delays = [f.expected_delay for f in mdp_filters]
#             print(expected_delays)
            route = routing_policy.choose_route(expected_delays)
#             print(route)
            delay = self.timeseries[route][t]
#             print(delay
            
            logbook.record({
                'hmm_filters': hmm_filters,
                'mdp_filters': mdp_filters,
                'action': action,
                'delay': delay,
                'route': route,
                'processing_time': processing_time,
#                 agrego el estado para saber en que nivel estuve
#                 'state': 
#                 o quiero el valor del rtt en ese estado?
#                 'state': state,
            })

        return logbook
    

# Helpers


def read_model(path):
    with open(path) as f:
        return HMM.from_dict(json.load(f))


def read_timeseries(path, fillna=False):
    ts = pd.read_csv(path).iloc[:, 1]
    if fillna:
        ts.fillna(method="ffill", inplace=True)
        ts.fillna(method="bfill", inplace=True)
    return ts.values
