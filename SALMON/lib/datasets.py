from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class NLNOGDataset:
    nodes: Dict[str, int]
    matrix: np.array

    @classmethod
    def read_nodes(cls, path):
        with open(path) as f:
            lines = f.readlines()
            nodes = filter(None, map(str.strip, lines[1:]))
        return {node: i for i, node in enumerate(nodes)}

    @classmethod
    def read_matrix(cls, path):
        matrix = np.loadtxt(path)
        matrix[matrix == 2000.0] = np.nan
        return matrix

    @classmethod
    def from_file(cls, nodes_path, matrix_path):
        nodes = cls.read_nodes(nodes_path)
        matrix = cls.read_matrix(matrix_path)
        return cls(nodes=nodes, matrix=matrix)

    def get_rtt(self, node_a, node_b):
        i = self.nodes[node_a]
        j = self.nodes[node_b]
        idxs = np.arange(i, self.matrix.shape[0], self.matrix.shape[1])
        return self.matrix[idxs, j]

    @property
    def pairs(self):
        return filter(lambda x: x[0] != x[1], product(self.nodes, repeat=2))
