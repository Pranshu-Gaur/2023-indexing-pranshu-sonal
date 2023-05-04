import faiss
import numpy as np
import time
import pandas as pd
from tqdm.auto import trange
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

d = 128  # vector size
M = 32
efSearch = 32  # number of entry points (neighbors) we use on each layer
efConstruction = 32  # number of entry points used on each layer
                     # during construction

index = faiss.IndexHNSWFlat(d, M)
print(index.hnsw)
levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)


def set_default_probas(M: int, m_L: float):
    nn = 0  # set nearest neighbors count = 0
    cum_nneighbor_per_level = []
    level = 0  # we start at level 0
    assign_probas = []
    while True:
        # calculate probability for current level
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        # once we reach low prob threshold, we've created enough levels
        if proba < 1e-9: break
        assign_probas.append(proba)
        # neighbors is == M on every level except level 0 where == M*2
        nn += M*2 if level == 0 else M
        cum_nneighbor_per_level.append(nn)
        level += 1
    return assign_probas, cum_nneighbor_per_level

assign_probas, cum_nneighbor_per_level = set_default_probas(
    32, 1/np.log(32)
)

def random_level(assign_probas: list, rng):
    # get random float from 'r'andom 'n'umber 'g'enerator
    f = rng.uniform() 
    for level in range(len(assign_probas)):
        # if the random float is less than level probability...
        if f < assign_probas[level]:
            # ... we assert at this level
            return level
        # otherwise subtract level probability and try again
        f -= assign_probas[level]
    # below happens with very low probability
    return len(assign_probas) - 1

import os

def get_memory(index):
    faiss.write_index(index, './temp.index')
    file_size = os.path.getsize('./temp.index')
    os.remove('./temp.index')
    return file_size

results = pd.DataFrame({
    'M': [],
    'efConstruction': [],
    'efSearch': [],
    'build_time': [],
    'search_time': [],
    'memory_usage': []
})
def HNSW(points, queries):
    M_bit = 5
    ef_bit = 5
    ef_search = {8}
    for i in trange(1, 2):
        M = 2 ** M_bit
        for j in trange(1, 2):
            efConstruction = 2 ** ef_bit
            index = faiss.IndexHNSWFlat(d, M)
            index.efConstruction = efConstruction
            start = time.perf_counter()
            index.add(points)
            build_time = (time.perf_counter() - start)
            memory_usage = get_memory(index)
            for efSearch in ef_search:
                index.efSearch = efSearch
                start = time.perf_counter()
                D, I = index.search(queries, k=1)
                search_time = (time.perf_counter() - start)
                results = results.append({
                    'M': M,
                    'efConstruction': efConstruction,
                    'efSearch': efSearch,
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_usage': memory_usage
                }, ignore_index=True)
            del index
    print(results)

