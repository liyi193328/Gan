
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
import numpy as np
import codecs


path = r"/home/bigdata/cwl/data_preprocessed/drop80.csv"
data = pd.read_csv(path)
print("read from {} done".format(path))

data = data[data.columns[1:]]

data = data.transpose()

data_val = data.values
del data

row_sum = np.sum(data_val, axis=1)
row_sum = np.expand_dims(row_sum, 1)

div = np.divide(data_val , row_sum)

del data_val
del row_sum

# mask = np.ma.masked_where(data_val == 0.0, data_val)
# data_f = np.multiply(np.log(div), 1 - mask.mask)

train_path = r"/home/bigdata/cwl/data_preprocessed/drop80.train"
infer_path = r"/home/bigdata/cwl/data_preprocessed/drop80.infer"
infer_data = div[-1600:,:]
train_data = div[0:-1600]
pd.DataFrame(train_data).to_csv(train_path, index=False)
print("data {} save to {}".format(train_data.shape, train_path))
pd.DataFrame(infer_data).to_csv(infer_path, index=False)
print("data {} save to {}".format(infer_data.shape, infer_path))


