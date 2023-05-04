import scann

import numpy as np
from numpy.linalg import norm

import time

def normalize_dataset(dataset):
  normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
  return normalized_dataset

def build_scann_searcher(dataset, queries):
  k = [1,2,4,8]
  print ('leaf\ttouch\tbuild_t\tsearch_t')
  for i in k:
    for j in range(1,11):
      s1 = time.process_time()
      searcher = scann.scann_ops_pybind.builder(normalize_dataset(dataset), 10, "dot_product").tree(
            num_leaves=1000*i, num_leaves_to_search=100*j, training_sample_size=1000000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(100).build()
      e1 = time.process_time()
      s2 = time.process_time()
      neighbours, distances = searcher.search_batched(queries)
      e2 = time.process_time()
      print ("{0}\t{1}\t{2:.6f}\t{3:.6f}".format(1000*i,100*j,float(e1-s1),float(e2-s2)/queries.shape[0]))

def ScaNN_Algo(dataset,queries):
  build_scann_searcher(dataset,queries)




