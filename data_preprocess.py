
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import codecs


# train_path = r"/home/bigdata/cwl/data_preprocessed/reprocess.csv"
# infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"

def reverse_normalization(data, factor=1e5):
  exp = np.exp(data) - 1
  row_sum = np.sum(exp, axis=1)
  row_sum = np.expand_dims(row_sum, 1)
  div = np.divide(exp, row_sum)
  div = np.log(1 + factor * div)
  return div

def row_normalization(data, factor=1e5):
  row_sum = np.sum(data, axis=1)
  row_sum = np.expand_dims(row_sum, 1)

  div = np.divide(data, row_sum)
  print("begin to loop cal log...")
  m, n = np.shape(div)
  div = np.log(1 + factor * div)

  return div

def divide_max(data):
  matrix_max = np.max(data)
  trans = np.divide(data, matrix_max)
  return trans

def log(data):
  return np.log(data + 1.0)

def same(data):
  return data

trans_map = {
  "reverse":reverse_normalization,
  "row_normal":row_normalization,
  "div_max": divide_max,
  "same": same,
  "log": log
}

def sub_handle(path, way,ind_col=0, trans=True ,save_path=None,**kwargs):
  data = pd.read_csv(path, header=0, sep=",", index_col=ind_col)
  print("read from {} done".format(path))
  if trans:
    data = data.transpose()
  columns = list(data.columns)
  print("{} data_shape is {}".format(path, data.shape))

  data = data.values
  data = trans_map[way](data,**kwargs)
  data = pd.DataFrame(data, columns=columns)
  print(data.shape)
  if save_path is not None:
    data.to_csv(save_path, index=False)
    print("saved to {}".format(save_path))
  return data

def handle_data(train_path, test_path, save_train_path, save_test_path, way = "div_max", **kwargs):
  train_df = sub_handle(train_path, way, **kwargs)
  test_df = sub_handle(test_path, way, **kwargs)

  all_df = pd.concat([train_df, test_df], axis=0)

  all_df.to_csv(save_train_path, index=False)
  print("save to {}".format(save_train_path))
  test_df.to_csv(save_test_path, index=False)
  print("save to {}".format(save_test_path))


if __name__ == "__main__":

  # train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop80.csv"
  # infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop80.csv"
  #
  # handle_data(train_path, infer_path, r"/home/bigdata/cwl/Gan/data/drop80_log.train", r"/home/bigdata/cwl/Gan/data/drop80_log.infer", way="log")

  # train_path = r"/home/bigdata/cwl/data_preprocessed/train_drop60.csv"
  # infer_path = r"/home/bigdata/cwl/data_preprocessed/test_drop60.csv"
  #
  # handle_data(train_path, infer_path, r"/home/bigdata/cwl/Gan/data/drop60_log.train",
  #             r"/home/bigdata/cwl/Gan/data/drop60_log.infer", way="log")


  # sub_handle("/home/bigdata/cwl/Gan/brain/brain_10.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/brain/brain_10.train",factor=1e6)

  sub_handle("/home/bigdata/cwl/Gan/chu/chu_sc.csv", "row_normal", save_path="/home/bigdata/cwl/Gan/data/chu/chu_sc.train",factor=1e6)