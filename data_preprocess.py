
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import codecs


train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop80.csv"
infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"

def handle_data(path, save_path):

  data = pd.read_csv(path, header=0, sep=",", index_col=0)
  print("read from {} done".format(path))
  data = data.transpose()
  print("{} data_shape is {}".format(path, data.shape))

  data_val = data.values
  del data
  row_sum = np.sum(data_val, axis=1)
  row_sum = np.expand_dims(row_sum, 1)

  div = np.divide(data_val , row_sum)
  m, n = np.shape(div)
  print("begin to loop cal log...")

  # for i in range(m):
  #   for j in range(n):
  #     if div[i][j] != 0:
  #       div[i][j] = 2.0 * div[i][j] - 1.0

  # div = 2.0 * div - 1.0
  pd.DataFrame(div).to_csv(save_path, index=False)

  print("save to {}".format(save_path))

if __name__ == "__main__":
  handle_data(train_path, r"/home/bigdata/cwl/Gan/data/drop80-0-1.train")
  handle_data(infer_path, r"/home/bigdata/cwl/Gan/data/drop80-0-1.infer")





