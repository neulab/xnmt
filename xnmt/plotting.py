from typing import Optional, Sequence, Union
import numbers
from unidecode import unidecode
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xnmt import utils

def plot_attention(src_words: Union[np.ndarray, Sequence[str]],
                   trg_words: Sequence[str],
                   attention_matrix: np.ndarray,
                   file_name: str,
                   size_x: numbers.Real = 8.0,
                   size_y: numbers.Real = 8.0) -> None:
  """This takes in source and target words and an attention matrix (in numpy format)
  and prints a visualization of this to a file.

  Args:
    src_words: a list of words in the source; alternatively, a numpy array containing speech features.
    trg_words: a list of target words
    attention_matrix: a two-dimensional numpy array of values between zero and one,
      where rows correspond to source words, and columns correspond to target words
    file_name: the name of the file to which we write the attention
    size_x: width of the main plot
    size_y: height of the plot
  """
  trg_words = [unidecode(w) for w in trg_words]
  src_is_speech = isinstance(src_words, np.ndarray)
  max_len = len(''.join(trg_words))
  if not src_is_speech:
    max_len = max(max_len, len(''.join(src_words)))
    src_words = [unidecode(w) for w in src_words]
  if max_len>150: matplotlib.rc('font', size=5)
  elif max_len>50: matplotlib.rc('font', size=7)
  dpi = 100 if max_len <= 150 else 150
  fig, axs = plt.subplots(nrows=1, ncols=2 if src_is_speech else 1,
                          figsize=(size_x+(1.0 if src_is_speech else 0.0), size_y),
                          gridspec_kw = {'width_ratios':[1, size_x]} if src_is_speech else None)
  ax = axs[1] if src_is_speech else axs
  # put the major ticks at the middle of each cell
  ax.set_xticks(np.arange(attention_matrix.shape[1]) + 0.5, minor=False)
  ax.set_yticks(np.arange(attention_matrix.shape[0]) + 0.5, minor=False)
  ax.invert_yaxis()
  if src_is_speech: plt.yticks([], [])

  # label axes by words
  ax.set_xticklabels(trg_words, minor=False)
  if not src_is_speech: ax.set_yticklabels(src_words, minor=False)
  ax.xaxis.tick_top()

  # draw the heatmap
  plt.pcolor(attention_matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)
  plt.colorbar()

  if src_is_speech:
    ax = axs[0]
    plot_speech_features(feature_matrix=src_words, ax=ax, dpi=dpi)
    fig.tight_layout()

  utils.make_parent_dir(file_name)
  plt.savefig(file_name, dpi=dpi)
  plt.close()


def plot_speech_features(feature_matrix: np.ndarray,
                         file_name: Optional[str] = None,
                         vertical: bool = True,
                         ax: Optional[matplotlib.axes.Axes] = None,
                         length: numbers.Real = 8.0,
                         dpi: numbers.Number = 100):
  """Plot speech feature matrix.

  Args:
    feature_matrix: a two-dimensional numpy array of values between zero and one,
      where rows correspond to source words, and columns correspond to target words
    file_name: the name of the file to which we write the attention; if not given, the plt context will be left un-closed
    vertical: if True, the time dimension will be projected onto the y axis, otherwise the x axis
    ax: if given, draw on this matplotlib axis; otherwise create a new figure
    length: figure length (if ax is not given)
    dpi: plot resolution
  """
  if not ax:
    plt.subplots(figsize=(1.0, length))
  if vertical: feature_matrix = feature_matrix[:,::-1].T
  if ax:
    ax.pcolor(feature_matrix, cmap=plt.cm.jet, vmin=-1, vmax=1)
    ax.axis('off')
  else:
    plt.pcolor(feature_matrix, cmap=plt.cm.jet, vmin=-1, vmax=1)
    plt.axis('off')
  if file_name is not None:
    utils.make_parent_dir(file_name)
    plt.savefig(file_name, dpi=dpi)
    plt.close()

