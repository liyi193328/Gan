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
  # Two subplots, the axes array is 1-d
  feature_indexs = [3, 100, 500, 1000, 4000, 6000, 8000, 10000]
  columns = list(sample_df.columns)

  for feature_index in feature_indexs:
    column = columns[feature_index]
    f, axarr = plt.subplots(2, sharex=True)
    plt.xlim(-1, len(infer_df) + 1)
    axarr[0].scatter(sample_df.index, sample_df[column])
    axarr[0].set_title('test: {}'.format(column))
    axarr[1].set_title("infer: {}".format(column))
    axarr[1].scatter(infer_df.index, infer_df[column])
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
  plot_save("/home/bigdata/cwl/Gan/samples/log_sigmoid/log_sigmoid.sample", "/home/bigdata/cwl/Gan/samples/log_sigmoid//log_sigmoid.2400", "/home/bigdata/cwl/Gan/samples/test.pdf")
