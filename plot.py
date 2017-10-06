#encoding=utf-8

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import codecs
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from matplotlib.backends.backend_pdf import PdfPages
import data_preprocess
import seaborn as sns
from scipy.stats import pearsonr

def get_top_feature(df, top=100):
  vars = np.var(df.values, axis=0)
  standard_vars = np.sqrt(vars)
  mean = np.mean(df.values, axis=0)
  x = np.divide(standard_vars, mean)
  top_vars_index = x.argsort()[-top:][::-1]
  return df.columns[top_vars_index]

def raw_dataframe(test_path_or_df, infer_path_or_df):
  if type(test_path_or_df) == str:
    dropout_df = data_preprocess.sub_handle(test_path_or_df,way="reverse",ind_col=None, trans=False)
    infer_df = data_preprocess.sub_handle(infer_path_or_df,way="reverse",ind_col=None,trans=False)
  else:
    dropout_df = test_path_or_df
    infer_df = infer_path_or_df

  magic_df = data_preprocess.sub_handle("/home/bigdata/cwl/Gan/prediction/drop80_magic.csv", way="row_normal",ind_col=None, trans=False)
  whole_df = data_preprocess.sub_handle("/home/bigdata/cwl/data_preprocessed/process_whole_test_drop80.csv", way="row_normal", ind_col=0, trans=True)
  scimpute_df = data_preprocess.sub_handle("/home/bigdata/scimpute_count.csv", way="row_normal", ind_col=0, trans=True)

  return [whole_df, dropout_df, magic_df, scimpute_df, infer_df]

def get_dataframe(test_path_or_df, infer_path_or_df):

  [whole_df, dropout_df, magic_df, scimpute_df, infer_df] = raw_dataframe(test_path_or_df, infer_path_or_df)

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

  return [whole_df, dropout_df, magic_df, scimpute_df, infer_df, indexs]

def get_similarity(test_path_or_df, infer_path_or_df):
  df_list = raw_dataframe(test_path_or_df, infer_path_or_df)
  [whole_df, dropout_df, magic_df, scimpute_df, infer_df] = df_list
  name_list = ["whole", "dropout", "magic", "scimpute", "ae"]
  for i in range(1, len(name_list)):
    print(name_list[i] + "...")
    df = df_list[i]
    x = np.reshape(df_list[0].values, (-1))
    y = np.reshape(df.values, (-1))
    coefficient, p_value = pearsonr(x, y)
    print(coefficient, p_value)
  return

def plot_headmap(test_path_or_df, infer_path_or_df, save_path, top_features=100, **kwargs):

  [whole_df, dropout_df, magic_df, scimpute_df, infer_df, indexs] = get_dataframe(test_path_or_df, infer_path_or_df)
  df_list = [whole_df, dropout_df, magic_df, scimpute_df, infer_df]
  name_list = ["whole", "dropout", "magic", "scimpute", "ae"]
  high_variance_columns = get_top_feature(whole_df)
  sel_df_list = [df[high_variance_columns] for df in df_list]
  plt.rcParams["figure.figsize"] = [20, 10]
  fig, axn = plt.subplots(1, 5, sharex=True, sharey=True)

  ax_list = list(axn.flat)
  # cbar_ax = fig.add_axes([.91, .3, .03, .4])
  for i in range(len(ax_list)):
    ax, name, df = ax_list[i], name_list[i], sel_df_list[i]
    ax.set_title(name)
    sns.heatmap(df, ax=ax, cmap="YlGnBu", xticklabels=False, yticklabels=False, cbar=False)
    #     , cbar= i == len(ax_list) - 1)

  fig.savefig(save_path, dpi=600)
  # fig.savefig(save_path.replace(".svg", ".png"), dpi=600)

def scatter_compare(test_path_or_df, infer_path_or_df, save_path=None, feature_choose_nums = 200):

  [whole_df, dropout_df, magic_df, scimpute_df, infer_df, indexs] = get_dataframe(test_path_or_df, infer_path_or_df)
  df_list = [whole_df, dropout_df, magic_df, scimpute_df, infer_df]
  name_list = ["whole", "dropout", "magic", "scimpute", "ae"]
  # feature_indexs = [3, 100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
  feature_indexs = np.random.choice(range(len(df_list[0])), feature_choose_nums)
  sub_df_list = [df[df.columns[feature_indexs]] for df in df_list]
  fig, axn = plt.subplots(1, 4, sharey=True)
  ax_list = axn.flat
  for i in range(1, len(df_list)):
    x = np.reshape(sub_df_list[0].values, (-1))
    y = np.reshape(sub_df_list[i].values, (-1))
    subplot = ax_list[i - 1]
    subplot.set_ylim(-0.5, np.max(y) + 0.5)
    subplot.set_xlim(-0.5, np.max(x) + 0.5)
    subplot.set_title(name_list[i])
    subplot.scatter(x, y)
  if save_path is not None:
    plt.savefig(save_path, dpi=600)

# def cal_corrcoef()
def plot_complete(test_path_or_df, infer_path_or_df, save_path):

  [whole_df, dropout_df, magic_df, scimpute_df, infer_df, indexs] = get_dataframe(test_path_or_df, infer_path_or_df)

  df_list = [whole_df, dropout_df, magic_df, scimpute_df, infer_df]
  name_list = ["whole", "dropout", "magic", "scimpute", "ae"]

  print(dropout_df.shape, infer_df.shape, magic_df.shape, scimpute_df.shape)
  # Two subplots, the axes array is 1-d
  feature_indexs = [3, 100, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
  feature_index = [3, 100]

  columns = list(dropout_df.columns)
  x = np.arange(0, len(indexs), 1)
  for feature_index in feature_indexs:

    column = columns[feature_index]

    f, axarr = plt.subplots(2,3,sharex=True)
    ax_list = axarr.flat

    series_list = [df[ df.columns[feature_index]] for df in df_list]

    whole_series, dropout_series, magic_series, scimpute_series, infer_series = series_list

    y_max = 0.0
    for y in series_list:
      y_max = max(y_max, y.max())
    y_max += 0.5

    greater_zero_index = np.where(dropout_series > 0.0)[0]
    equal_zero_index = np.where(dropout_series == 0.0)[0]

    for i in range(len(name_list)):
      ax = ax_list[i]
      ax.set_title("{}: {}".format(name_list[i], column))
      ax.set_xlim(-1, len(infer_df) + 1)
      ax.set_ylim(0, y_max)
      for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(5)
      sizes = 8 ** 2
      if name_list[i] in ["magic", "scimpute", "ae"]:
        ax.scatter(greater_zero_index, series_list[i].iloc[greater_zero_index], s=sizes)
        ax.scatter(equal_zero_index, series_list[i].iloc[equal_zero_index], color="r", s=sizes)
      else:
        ax.scatter(x, series_list[i], s=sizes)

    # axarr[0,0].set_title("whole: {}".format(column))
    # axarr[0, 0].
    # axarr[0,0].scatter(x, whole_series)
    #
    # axarr[0,1].set_title('dropout: {}'.format(column))
    # axarr[0, 1].set_ylim(0, y_max)
    # axarr[0,1].scatter(x, dropout_series)
    #
    # axarr[1,0].set_title('magic: {}'.format(column))
    # axarr[1,0].set_ylim(0, y_max)
    # axarr[1,0].scatter(greater_zero_index, magic_series.iloc[greater_zero_index])
    # axarr[1,0].scatter(equal_zero_index, magic_series.iloc[equal_zero_index], color="r")
    # # axarr[1,0].scatter(x, magic_df[column])
    #
    # axarr[1,1].set_title("scimpute:{}".format(column))
    # axarr[1,1].set_ylim(0, y_max)
    #
    #
    # axarr[0,2].set_title("ae: {}".format(column))
    # axarr[0,2].set_ylim(0, y_max)
    # axarr[0,2].scatter(greater_zero_index, infer_series.iloc[greater_zero_index])
    # axarr[0,2].scatter(equal_zero_index, infer_series.iloc[equal_zero_index], color="r")


    cur_fig_path = save_path.replace(".png", "{}.png".format(feature_index))
    f.savefig(cur_fig_path, dpi=600)

  # pp = PdfPages(save_path)
  # figs = None
  # if figs is None:
  #   figs = [plt.figure(n) for n in plt.get_fignums()]
  # for fig in figs:
  #   fig.savefig(pp, format='pdf', dpi=600)
  # pp.close()
  # print("saved fig pdfs to {}".format(save_path))


if __name__ == "__main__":

  plot_complete("/home/bigdata/cwl/Gan/data/drop80_log.infer", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/log_sigmoid.3500.infer.complete", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/complete.png")
  # plot_headmap("/home/bigdata/cwl/Gan/data/drop80_log.infer", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/log_sigmoid.3500.infer.complete", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/headmap.png")
  # get_similarity("/home/bigdata/cwl/Gan/data/drop80_log.infer", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/log_sigmoid.3500.infer.complete")
  # scatter_compare("/home/bigdata/cwl/Gan/data/drop80_log.infer", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/log_sigmoid.3500.infer.complete", "/home/bigdata/cwl/Gan/prediction/log_sigmoid/scatter_compare.png")
