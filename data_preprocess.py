
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import codecs


train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop80.csv"
infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"

def row_normalization(data):
  row_sum = np.sum(data, axis=1)
  row_sum = np.expand_dims(row_sum, 1)

  div = np.divide(data, row_sum)
  print("begin to loop cal log...")
  m, n = np.shape(div)
  for i in range(m):
    for j in range(n):
      if div[i][j] != 0:
        div[i][j] = 2.0 * div[i][j] - 1.0

  return div

def divide_max(data):
  matrix_max = np.max(data)
  trans = np.divide(data, matrix_max)
  return trans

def same(data):
  return data

trans_map = {
  "row_normal":row_normalization,
  "div_max": divide_max,
  "same": same
}

def handle_data(path, save_path, way = "div_max"):

  data = pd.read_csv(path, header=0, sep=",", index_col=0)
  print("read from {} done".format(path))
  data = data.transpose()
  print("{} data_shape is {}".format(path, data.shape))

  data = data.values
  data = trans_map[way](data)
  pd.DataFrame(data).to_csv(save_path, index=False)

  print("save to {}".format(save_path))

if __name__ == "__main__":
  handle_data(train_path, r"/home/bigdata/cwl/Gan/data/drop80_or.train", way="same")
  handle_data(infer_path, r"/home/bigdata/cwl/Gan/data/drop80_or.infer", way="same")





