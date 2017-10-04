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
import data_preprocess

def plot_save(test_path_or_df, infer_path_or_df, save_path):
  if type(test_path_or_df) == str:
    dropout_df = data_preprocess.sub_handle(test_path_or_df,way="reverse",ind_col=None, trans=False)
    infer_df = data_preprocess.sub_handle(infer_path_or_df,way="reverse",ind_col=None,trans=False)
  else:
    dropout_df = test_path_or_df
    infer_df = infer_path_or_df

  magic_df = data_preprocess.sub_handle("/home/bigdata/cwl/Gan/prediction/drop80_magic.csv", way="row_normal",ind_col=None, trans=False)
  whole_df = data_preprocess.sub_handle("/home/bigdata/cwl/data_preprocessed/process_whole_test_drop80.csv", way="row_normal", ind_col=0, trans=True)
  scimpute_df = data_preprocess.sub_handle("/home/bigdata/scimpute_count.csv", way="row_normal", ind_col=0, trans=True)

  # or_df = data_preprocess.sub_handle("/home/bigdata/cwl/data_preprocessed/expx_dropprocessed.csv", way="row_normal",
  #                                    trans=False)

  base_index = 2
  indexs = np.arange(base_index, len(dropout_df), 10)
  # indexs = np.arange(0, len(dropout_df), 1)

  dropout_df = dropout_df.iloc[indexs]
  infer_df = infer_df.iloc[indexs]
  magic_df = magic_df.iloc[indexs]
  whole_df = whole_df.iloc[indexs]
  scimpute_df = scimpute_df.iloc[indexs]
  dropout_df = whole_df * np.float32(dropout_df > 0.0)

  print(dropout_df.shape, infer_df.shape, magic_df.shape, scimpute_df.shape)
  # Two subplots, the axes array is 1-d
  feature_indexs = [3, 100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
  columns = list(dropout_df.columns)
  x = np.arange(0, len(indexs), 1)
  for feature_index in feature_indexs:

    column = columns[feature_index]

    f, axarr = plt.subplots(2,3,sharex=True)
    plt.xlim(-1, len(infer_df) + 1)

    whole_series = whole_df[whole_df.columns[feature_index]]
    dropout_series = dropout_df[column]
    magic_series = magic_df[column]
    scimpute_series = scimpute_df[column]
    infer_series = infer_df[column]

    y_max = 0.0
    for y in [whole_series, dropout_series, magic_series, scimpute_series, infer_series]:
      y_max = max(y_max, y.max())
    y_max += 0.5

    greater_zero_index = np.where(dropout_series > 0.0)[0]
    equal_zero_index = np.where(dropout_series == 0.0)[0]

    axarr[0,0].set_title("whole: {}".format(column))
    axarr[0, 0].set_ylim(0, y_max)
    axarr[0,0].scatter(x, whole_series)

    axarr[0,1].set_title('dropout: {}'.format(column))
    axarr[0, 1].set_ylim(0, y_max)
    axarr[0,1].scatter(x, dropout_series)

    axarr[1,0].set_title('magic: {}'.format(column))
    axarr[1,0].set_ylim(0, y_max)
    axarr[1,0].scatter(greater_zero_index, magic_series.iloc[greater_zero_index])
    axarr[1,0].scatter(equal_zero_index, magic_series.iloc[equal_zero_index], color="r")
    # axarr[1,0].scatter(x, magic_df[column])

    axarr[1, 1].set_title("AE: {}".format(column))
    axarr[1, 1].set_ylim(0, y_max)
    axarr[1,1].scatter(greater_zero_index, infer_series.iloc[greater_zero_index])
    axarr[1,1].scatter(equal_zero_index, infer_series.iloc[equal_zero_index], color="r")

    axarr[0,2].set_title("scimpute:{}".format(column))
    axarr[0,2].set_ylim(0, y_max)
    axarr[0,2].scatter(greater_zero_index, scimpute_series.iloc[greater_zero_index])
    axarr[0,2].scatter(equal_zero_index, scimpute_series.iloc[equal_zero_index], color="r")

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
