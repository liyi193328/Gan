#encoding=utf-8

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import codecs
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_save(test_path_or_df, infer_path_or_df, save_path):
  if type(test_path_or_df) == str:
    sample_df = pd.read_csv(test_path_or_df)
    infer_df = pd.read_csv(infer_path_or_df)
  else:
    sample_df = test_path_or_df
    infer_df = infer_path_or_df

  magic_df = pd.read_csv("/home/bigdata/cwl/Gan/prediction/drop80_magic.csv", header=None)
  base_index = 2

  # print("len sample_df: {}".format(len(sample_df)))

  indexs = np.arange(base_index, len(sample_df), 10)
  sample_df = sample_df.iloc[indexs]
  infer_df = infer_df.iloc[indexs]
  magic_df = magic_df.iloc[indexs]

  or_df = pd.read_csv("/home/bigdata/cwl/data_preprocessed/expx_dropprocessed.csv", header=0, sep=",", index_col=0)
  magic_df.columns = or_df.columns
  # print(sample_df.shape, infer_df.shape, or_df.shape)
  # Two subplots, the axes array is 1-d
  feature_indexs = [3, 100, 500, 1000, 4000, 6000, 8000, 10000]
  columns = list(sample_df.columns)
  x = np.arange(0, len(indexs), 1)
  for feature_index in feature_indexs:
    column = columns[feature_index]
    f, axarr = plt.subplots(2,2,sharex=True)
    plt.xlim(-1, len(infer_df) + 1)
    axarr[0,0].set_title("original: {}".format(column))
    axarr[0,0].scatter(x, or_df[or_df.columns[feature_index]])
    axarr[0,1].set_title('dropout: {}'.format(column))
    axarr[0,1].scatter(x, sample_df[column])
    axarr[1,0].set_title('magic: {}'.format(column))
    axarr[1,0].scatter(x, magic_df[column])
    axarr[1,1].set_title("autoencoder: {}".format(column))
    axarr[1,1].scatter(x, infer_df[column])

  # print(plt.get_fignums())

  pp = PdfPages(save_path)
  figs = None
  if figs is None:
    figs = [plt.figure(n) for n in plt.get_fignums()]
  for fig in figs:
    fig.savefig(pp, format='pdf')

  pp.close()
  print("saved fig pdfs to {}".format(save_path))

if __name__ == "__main__":
  plot_save("/home/bigdata/cwl/Gan/data/drop80_log.infer", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/log_sigmoid.3500.infer.complete", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/test.pdf")
