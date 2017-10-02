#encoding=utf-8

import os
import codecs
import numpy as np
import pandas as pd

def plot(test_df, infer_df, columns, step):
  feature_indexs = [3, 100, 500, 1000, 4000, 6000, 8000, 100000]

