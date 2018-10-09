"""
This module implements self-attentional acoustic models as described in the following paper:

Self-Attentional Acoustic Models
Matthias Sperber, Jan Niehues, Graham Neubig, Sebastian Stüker, Alex Waibel
Interspeech 2018
https://arxiv.org/abs/1803.09519

The main class to be aware of is :class:`SAAMSeqTransducer`.
"""

import typing
import logging
import numbers

yaml_logger = logging.getLogger('yaml')
from collections.abc import Sequence
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy

import numpy as np
import dynet as dy

import xnmt.param_initializers
from xnmt import norms, events, param_collections
from xnmt.modelparts import transforms
from xnmt.transducers import recurrent, base as transducers
from xnmt.expression_seqs import ExpressionSequence
from xnmt.persistence import Serializable, serializable_init, Ref, bare
from xnmt.modelparts import embedders

LOG_ATTENTION = False


class SAAMTimeDistributed(object):
  """
  A Callable that puts the time-dimension of an input expression into the batch dimension via a reshape.
  """

  def __call__(self, x: dy.Expression) -> dy.Expression:
    """
    Move the time-dimension of an input expression into the batch dimension via a reshape.

    Args:
      x: expression of dimensions ((hidden, timesteps), batch_size)

    Returns:
      expression of dimensions ((hidden,), timesteps*batch_size)
    """
    batch_size = x[0].dim()[1]
    model_dim = x[0].dim()[0][0]
    seq_len = len(x)
    total_words = seq_len * batch_size
    input_tensor = x.as_tensor()
    return dy.reshape(input_tensor, (model_dim,), batch_size=total_words)


class SAAMPositionwiseFeedForward(Serializable):
  """
  Interleaved feed-forward components of the transformer, computed as layer_norm(dropout(linear(relu(linear))) + x).

  Args:
    input_dim: the size of input for the first-layer of the FFN.
    hidden_dim: the hidden layer size of the second-layer of the FNN.
    nonlinearity: non-linearity, a DyNet unary operation (rectify according to the transformer paper)
    linear_transforms: list with 2 items corresponding to the first and second linear transformations to be used
                       (pass ``None`` to create automatically)
    layer_norm: layer norm object to be applied (pass ``None`` to create automatically)
  """
  yaml_tag = "!SAAMPositionwiseFeedForward"

  @serializable_init
  def __init__(self, input_dim: int, hidden_dim: int, nonlinearity: str = "rectify",
               linear_transforms: typing.Optional[typing.Sequence[transforms.Linear]] = None,
               layer_norm: typing.Optional[norms.LayerNorm] = None) -> None:
    w_12 = self.add_serializable_component("linear_transforms", linear_transforms,
                                           lambda: [transforms.Linear(input_dim, hidden_dim),
                                                    transforms.Linear(hidden_dim, input_dim)])
    self.w_1 = w_12[0]
    self.w_2 = w_12[1]
    self.layer_norm = self.add_serializable_component("layer_norm", layer_norm, lambda: norms.LayerNorm(input_dim))
    self.nonlinearity = getattr(dy, nonlinearity)

  def __call__(self, x, p):
    residual = x
    output = self.w_2.transform(self.nonlinearity(self.w_1.transform(x)))
    if p > 0.0:
      output = dy.dropout(output, p)
    return self.layer_norm.transform(output + residual)


class SAAMMultiHeadedSelfAttention(Serializable):
  """
  Args:
    head_count: number of self-att heads
    model_dim: model dimension
    downsample_factor: downsampling factor (>=1)
    input_dim: input dimension
    ignore_masks: don't apply any masking
    plot_attention: None or path to directory to write plots to
    diag_gauss_mask: False to disable, otherwise a float denoting the std of the mask
    square_mask_std: whether to square the std parameter to facilitate training
    cross_pos_encoding_type: None 'embedding'
    kq_pos_encoding_type: None or 'embedding'
    kq_pos_encoding_size:
    max_len: max sequence length (used to determine positional embedding size)
    param_init: initializer for weight matrices
    bias_init: initializer for bias vectors
    linear_kvq:
    kq_positional_embedder:
    layer_norm:
    res_shortcut:
    desc: useful to describe layer if plot_attention is given.
  """
  yaml_tag = "!SAAMMultiHeadedSelfAttention"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self, head_count: int, model_dim: int, downsample_factor: int = 1, input_dim: int = None,
               ignore_masks: bool = False, plot_attention: typing.Optional[str] = None,
               diag_gauss_mask: typing.Union[bool, numbers.Real] = False,
               square_mask_std: bool = True, cross_pos_encoding_type: typing.Optional[str] = None,
               kq_pos_encoding_type: typing.Optional[str] = None, kq_pos_encoding_size: int = 40, max_len: int = 1500,
               param_init: xnmt.param_initializers.ParamInitializer = xnmt.param_initializers.GlorotInitializer(),
               bias_init: xnmt.param_initializers.ParamInitializer = xnmt.param_initializers.ZeroInitializer(),
               linear_kvq = None, kq_positional_embedder = None, layer_norm = None, res_shortcut = None,
               desc: typing.Any = None) -> None:
    if input_dim is None: input_dim = model_dim
    self.input_dim = input_dim
    assert model_dim % head_count == 0
    self.dim_per_head = model_dim // head_count
    self.model_dim = model_dim
    self.head_count = head_count
    assert downsample_factor >= 1
    self.downsample_factor = downsample_factor
    self.plot_attention = plot_attention
    self.plot_attention_counter = 0
    self.desc = desc

    self.ignore_masks = ignore_masks
    self.diag_gauss_mask = diag_gauss_mask
    self.square_mask_std = square_mask_std

    self.kq_pos_encoding_type = kq_pos_encoding_type
    self.kq_pos_encoding_size = kq_pos_encoding_size
    self.max_len = max_len

    subcol = param_collections.ParamManager.my_params(self)

    if self.kq_pos_encoding_type is None:
      self.linear_kvq = self.add_serializable_component("linear_kvq", linear_kvq,
                                                        lambda: transforms.Linear(input_dim * downsample_factor,
                                                                                  head_count * self.dim_per_head * 3,
                                                                                  param_init=param_init,
                                                                                  bias_init=bias_init))
    else:
      self.linear_kq, self.linear_v = \
        self.add_serializable_component("linear_kvq",
                                        linear_kvq,
                                        lambda: [
                                          transforms.Linear(input_dim * downsample_factor + self.kq_pos_encoding_size,
                                                            head_count * self.dim_per_head * 2, param_init=param_init,
                                                            bias_init=bias_init),
                                          transforms.Linear(input_dim * downsample_factor, head_count * self.dim_per_head,
                                                            param_init=param_init, bias_init=bias_init)])
      assert self.kq_pos_encoding_type == "embedding"
      self.kq_positional_embedder = self.add_serializable_component("kq_positional_embedder",
                                                                    kq_positional_embedder,
                                                                    lambda: embedders.PositionEmbedder(
                                                                      max_pos=self.max_len,
                                                                      emb_dim=self.kq_pos_encoding_size,
                                                                      param_init=param_init))

    if self.diag_gauss_mask:
      if self.diag_gauss_mask == "rand":
        rand_init = np.exp((np.random.random(size=(self.head_count,))) * math.log(1000))
        self.diag_gauss_mask_sigma = subcol.add_parameters(dim=(1, 1, self.head_count),
                                                          init=dy.NumpyInitializer(rand_init))
      else:
        self.diag_gauss_mask_sigma = subcol.add_parameters(dim=(1, 1, self.head_count),
                                                          init=dy.ConstInitializer(self.diag_gauss_mask))

    self.layer_norm = self.add_serializable_component("layer_norm", layer_norm, lambda: norms.LayerNorm(model_dim))

    if model_dim != input_dim * downsample_factor:
      self.res_shortcut = self.add_serializable_component("res_shortcut", res_shortcut,
                                                          lambda: transforms.Linear(input_dim * downsample_factor,
                                                                                    model_dim,
                                                                                    param_init=param_init,
                                                                                    bias_init=bias_init))
    self.cross_pos_encoding_type = cross_pos_encoding_type
    if cross_pos_encoding_type == "embedding":
      self.cross_pos_emb_p1 = subcol.add_parameters(dim=(self.max_len, self.dim_per_head, self.head_count),
                                                   init=dy.NormalInitializer(mean=1.0, var=0.001))
      self.cross_pos_emb_p2 = subcol.add_parameters(dim=(self.max_len, self.dim_per_head, self.head_count),
                                                   init=dy.NormalInitializer(mean=1.0, var=0.001))
    elif cross_pos_encoding_type is not None:
      raise NotImplementedError()

  def plot_att_mat(self, mat, filename, dpi=1200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(mat)
    ax.set_aspect('auto')
    fig.savefig(filename, dpi=dpi)
    fig.clf()
    plt.close('all')

  def shape_projection(self, x, batch_size):
    total_words = x.dim()[1]
    seq_len = total_words / batch_size
    out = dy.reshape(x, (self.model_dim, seq_len), batch_size=batch_size)
    out = dy.transpose(out)
    return dy.reshape(out, (seq_len, self.dim_per_head), batch_size=batch_size * self.head_count)

  def __call__(self, x: dy.Expression, att_mask: np.ndarray, batch_mask: np.ndarray, p: numbers.Real):
    """
    x: expression of dimensions (input_dim, time) x batch
    att_mask: numpy array of dimensions (time, time); pre-transposed
    batch_mask: numpy array of dimensions (batch, time)
    p: dropout prob
    """
    sent_len = x.dim()[0][1]
    batch_size = x[0].dim()[1]

    if self.downsample_factor > 1:
      if sent_len % self.downsample_factor != 0:
        raise ValueError(
          "For 'reshape' downsampling, sequence lengths must be multiples of the downsampling factor. "
          "Configure batcher accordingly.")
      if batch_mask is not None: batch_mask = batch_mask[:, ::self.downsample_factor]
      sent_len_out = sent_len // self.downsample_factor
      sent_len = sent_len_out
      out_mask = x.mask
      if self.downsample_factor > 1 and out_mask is not None:
        out_mask = out_mask.lin_subsampled(reduce_factor=self.downsample_factor)

      x = ExpressionSequence(expr_tensor=dy.reshape(x.as_tensor(), (
        x.dim()[0][0] * self.downsample_factor, x.dim()[0][1] / self.downsample_factor), batch_size=batch_size),
                                                 mask=out_mask)
      residual = SAAMTimeDistributed()(x)
    else:
      residual = SAAMTimeDistributed()(x)
      sent_len_out = sent_len
    if self.model_dim != self.input_dim * self.downsample_factor:
      residual = self.res_shortcut.transform(residual)

    # Concatenate all the words together for doing vectorized affine transform
    if self.kq_pos_encoding_type is None:
      kvq_lin = self.linear_kvq.transform(SAAMTimeDistributed()(x))
      key_up = self.shape_projection(dy.pick_range(kvq_lin, 0, self.head_count * self.dim_per_head), batch_size)
      value_up = self.shape_projection(
        dy.pick_range(kvq_lin, self.head_count * self.dim_per_head, 2 * self.head_count * self.dim_per_head),
        batch_size)
      query_up = self.shape_projection(
        dy.pick_range(kvq_lin, 2 * self.head_count * self.dim_per_head, 3 * self.head_count * self.dim_per_head),
        batch_size)
    else:
      assert self.kq_pos_encoding_type == "embedding"
      encoding = self.kq_positional_embedder.embed_sent(sent_len).as_tensor()
      kq_lin = self.linear_kq.transform(
        SAAMTimeDistributed()(
          ExpressionSequence(expr_tensor=dy.concatenate([x.as_tensor(), encoding]))))
      key_up = self.shape_projection(dy.pick_range(kq_lin, 0, self.head_count * self.dim_per_head), batch_size)
      query_up = self.shape_projection(
        dy.pick_range(kq_lin, self.head_count * self.dim_per_head, 2 * self.head_count * self.dim_per_head), batch_size)
      v_lin = self.linear_v.transform(SAAMTimeDistributed()(x))
      value_up = self.shape_projection(v_lin, batch_size)

    if self.cross_pos_encoding_type:
      assert self.cross_pos_encoding_type == "embedding"
      emb1 = dy.pick_range(dy.parameter(self.cross_pos_emb_p1), 0, sent_len)
      emb2 = dy.pick_range(dy.parameter(self.cross_pos_emb_p2), 0, sent_len)
      key_up = dy.reshape(key_up, (sent_len, self.dim_per_head, self.head_count), batch_size=batch_size)
      key_up = dy.concatenate_cols([dy.cmult(key_up, emb1), dy.cmult(key_up, emb2)])
      key_up = dy.reshape(key_up, (sent_len, self.dim_per_head * 2), batch_size=self.head_count * batch_size)
      query_up = dy.reshape(query_up, (sent_len, self.dim_per_head, self.head_count), batch_size=batch_size)
      query_up = dy.concatenate_cols([dy.cmult(query_up, emb2), dy.cmult(query_up, -emb1)])
      query_up = dy.reshape(query_up, (sent_len, self.dim_per_head * 2), batch_size=self.head_count * batch_size)

    scaled = query_up * dy.transpose(
      key_up / math.sqrt(self.dim_per_head))  # scale before the matrix multiplication to save memory

    # Apply Mask here
    if not self.ignore_masks:
      if att_mask is not None:
        att_mask_inp = att_mask * -100.0
        if self.downsample_factor > 1:
          att_mask_inp = att_mask_inp[::self.downsample_factor, ::self.downsample_factor]
        scaled += dy.inputTensor(att_mask_inp)
      if batch_mask is not None:
        # reshape (batch, time) -> (time, head_count*batch), then *-100
        inp = np.resize(np.broadcast_to(batch_mask.T[:, np.newaxis, :],
                                        (sent_len, self.head_count, batch_size)),
                        (1, sent_len, self.head_count * batch_size)) \
              * -100
        mask_expr = dy.inputTensor(inp, batched=True)
        scaled += mask_expr
      if self.diag_gauss_mask:
        diag_growing = np.zeros((sent_len, sent_len, self.head_count))
        for i in range(sent_len):
          for j in range(sent_len):
            diag_growing[i, j, :] = -(i - j) ** 2 / 2.0
        e_diag_gauss_mask = dy.inputTensor(diag_growing)
        e_sigma = dy.parameter(self.diag_gauss_mask_sigma)
        if self.square_mask_std:
          e_sigma = dy.square(e_sigma)
        e_sigma_sq_inv = dy.cdiv(dy.ones(e_sigma.dim()[0], batch_size=batch_size), dy.square(e_sigma))
        e_diag_gauss_mask_final = dy.cmult(e_diag_gauss_mask, e_sigma_sq_inv)
        scaled += dy.reshape(e_diag_gauss_mask_final, (sent_len, sent_len), batch_size=batch_size * self.head_count)

    # Computing Softmax here.
    attn = dy.softmax(scaled, d=1)
    if LOG_ATTENTION:
      yaml_logger.info({"key": "selfatt_mat_ax0", "value": np.average(attn.value(), axis=0).dumps(), "desc": self.desc})
      yaml_logger.info({"key": "selfatt_mat_ax1", "value": np.average(attn.value(), axis=1).dumps(), "desc": self.desc})
      yaml_logger.info({"key": "selfatt_mat_ax0_ent", "value": entropy(attn.value()).dumps(), "desc": self.desc})
      yaml_logger.info(
        {"key": "selfatt_mat_ax1_ent", "value": entropy(attn.value().transpose()).dumps(), "desc": self.desc})

    self.select_att_head = 0
    if self.select_att_head is not None:
      attn = dy.reshape(attn, (sent_len, sent_len, self.head_count), batch_size=batch_size)
      sel_mask = np.zeros((1, 1, self.head_count))
      sel_mask[0, 0, self.select_att_head] = 1.0
      attn = dy.cmult(attn, dy.inputTensor(sel_mask))
      attn = dy.reshape(attn, (sent_len, sent_len), batch_size=self.head_count * batch_size)

    # Applying dropout to attention
    if p > 0.0:
      drop_attn = dy.dropout(attn, p)
    else:
      drop_attn = attn

    # Computing weighted attention score
    attn_prod = drop_attn * value_up

    # Reshaping the attn_prod to input query dimensions
    out = dy.reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), batch_size=batch_size)
    out = dy.transpose(out)
    out = dy.reshape(out, (self.model_dim,), batch_size=batch_size * sent_len_out)
    #     out = dy.reshape_transpose_reshape(attn_prod, (sent_len_out, self.dim_per_head * self.head_count), (self.model_dim,), pre_batch_size=batch_size, post_batch_size=batch_size*sent_len_out)

    if self.plot_attention:
      from sklearn.metrics.pairwise import cosine_similarity
      assert batch_size == 1
      mats = []
      for i in range(attn.dim()[1]):
        mats.append(dy.pick_batch_elem(attn, i).npvalue())
        self.plot_att_mat(mats[-1],
                          "{}.sent_{}.head_{}.png".format(self.plot_attention, self.plot_attention_counter, i),
                          300)
      avg_mat = np.average(mats, axis=0)
      self.plot_att_mat(avg_mat,
                        "{}.sent_{}.head_avg.png".format(self.plot_attention, self.plot_attention_counter),
                        300)
      cosim_before = cosine_similarity(x.as_tensor().npvalue().T)
      self.plot_att_mat(cosim_before,
                        "{}.sent_{}.cosim_before.png".format(self.plot_attention, self.plot_attention_counter),
                        600)
      cosim_after = cosine_similarity(out.npvalue().T)
      self.plot_att_mat(cosim_after,
                        "{}.sent_{}.cosim_after.png".format(self.plot_attention, self.plot_attention_counter),
                        600)
      self.plot_attention_counter += 1

    # Adding dropout and layer normalization
    if p > 0.0:
      res = dy.dropout(out, p) + residual
    else:
      res = out + residual
    ret = self.layer_norm.transform(res)
    return ret

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    yaml_logger.info(
      {"key": "self_att_mask_var: ", "val": [float(x) for x in list(self.diag_gauss_mask_sigma.as_array().flat)],
       "desc": self.desc})


class TransformerEncoderLayer(Serializable):
  yaml_tag = "!TransformerEncoderLayer"
  @serializable_init
  def __init__(self, hidden_dim, head_count=8, ff_hidden_dim=2048, downsample_factor=1,
               input_dim=None, diagonal_mask_width=None, ignore_masks=False,
               plot_attention=None, nonlinearity="rectify", diag_gauss_mask=False,
               square_mask_std=True, cross_pos_encoding_type=None,
               ff_lstm=False, kq_pos_encoding_type=None, kq_pos_encoding_size=40, max_len=1500,
               param_init=Ref("exp_global.param_init", default=bare(xnmt.param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(xnmt.param_initializers.ZeroInitializer)),
               dropout=None, self_attn=None, feed_forward=None, desc=None):
    self.self_attn = self.add_serializable_component("self_attn",
                                                     self_attn,
                                                     lambda: SAAMMultiHeadedSelfAttention(head_count, hidden_dim,
                                                                                          downsample_factor,
                                                                                          input_dim=input_dim,
                                                                                          ignore_masks=ignore_masks,
                                                                                          plot_attention=plot_attention,
                                                                                          diag_gauss_mask=diag_gauss_mask,
                                                                                          square_mask_std=square_mask_std,
                                                                                          param_init=param_init,
                                                                                          bias_init=bias_init,
                                                                                          cross_pos_encoding_type=cross_pos_encoding_type,
                                                                                          kq_pos_encoding_type=kq_pos_encoding_type,
                                                                                          kq_pos_encoding_size=kq_pos_encoding_size,
                                                                                          max_len=max_len,
                                                                                          desc=desc))
    self.ff_lstm = ff_lstm
    if ff_lstm:
      self.feed_forward = self.add_serializable_component("feed_forward",
                                                          feed_forward,
                                                          lambda: recurrent.BiLSTMSeqTransducer(layers=1,
                                                                                                input_dim=hidden_dim,
                                                                                                hidden_dim=hidden_dim,
                                                                                                dropout=dropout,
                                                                                                param_init=param_init,
                                                                                                bias_init=bias_init))
    else:
      self.feed_forward = self.add_serializable_component("feed_forward",
                                                          feed_forward,
                                                          lambda: SAAMPositionwiseFeedForward(hidden_dim, ff_hidden_dim,
                                                                                              nonlinearity=nonlinearity,
                                                                                              param_init=param_init))
    self.head_count = head_count
    self.downsample_factor = downsample_factor
    self.diagonal_mask_width = diagonal_mask_width
    if diagonal_mask_width: assert diagonal_mask_width % 2 == 1

  def set_dropout(self, dropout):
    self.dropout = dropout

  def transduce(self, x: ExpressionSequence) -> ExpressionSequence:
    seq_len = len(x)
    batch_size = x[0].dim()[1]

    att_mask = None
    if self.diagonal_mask_width is not None:
      if self.diagonal_mask_width is None:
        att_mask = np.zeros((seq_len, seq_len))
      else:
        att_mask = np.ones((seq_len, seq_len))
        for i in range(seq_len):
          from_i = max(0, i - self.diagonal_mask_width // 2)
          to_i = min(seq_len, i + self.diagonal_mask_width // 2 + 1)
          att_mask[from_i:to_i, from_i:to_i] = 0.0

    mid = self.self_attn(x=x, att_mask=att_mask, batch_mask=x.mask.np_arr if x.mask else None, p=self.dropout)
    if self.downsample_factor > 1:
      seq_len = int(math.ceil(seq_len / float(self.downsample_factor)))
    hidden_dim = mid.dim()[0][0]
    out_mask = x.mask
    if self.downsample_factor > 1 and out_mask is not None:
      out_mask = out_mask.lin_subsampled(reduce_factor=self.downsample_factor)
    if self.ff_lstm:
      mid_re = dy.reshape(mid, (hidden_dim, seq_len), batch_size=batch_size)
      out = self.feed_forward.transduce(ExpressionSequence(expr_tensor=mid_re, mask=out_mask))
      out = dy.reshape(out.as_tensor(), (hidden_dim,), batch_size=seq_len * batch_size)
    else:
      out = self.feed_forward.transduce(mid, p=self.dropout)

    self._recent_output = out
    return ExpressionSequence(
      expr_tensor=dy.reshape(out, (out.dim()[0][0], seq_len), batch_size=batch_size),
      mask=out_mask)


class SAAMSeqTransducer(transducers.SeqTransducer, Serializable):
  """
  Args:
    input_dim: input dimension
    layers: number of layers
    hidden_dim: hidden dimension
    head_count: number of self-attention heads
    ff_hidden_dim: hidden dimension of the interleaved feed-forward layers
    dropout: dropout probability
    downsample_factor: downsampling factor (>=1)
    diagonal_mask_width: if given, apply hard masking of this width
    ignore_masks: if True, don't apply any masking
    plot_attention: if given, plot self-attention matrices for each layer
    nonlinearity: nonlinearity to apply in FF module (string that corresponds to a DyNet unary operation)
    pos_encoding_type: None, trigonometric, embedding
    pos_encoding_combine: add, concat
    pos_encoding_size: if add, must eqal input_dim, otherwise can be chosen freely
    max_len: needed to initialize pos embeddings
    diag_gauss_mask: whether to apply a soft Gaussian mask with learnable variance
    square_mask_std: initial standard deviation of soft Gaussian mask
    cross_pos_encoding_type: 'embedding' or None
    ff_lstm: if True, use interleaved LSTMs, otherwise add depth using position-wise feed-forward components
    kq_pos_encoding_type: None or 'embedding'
    kq_pos_encoding_size (int): size of position embeddings
    param_init: initializer for weight matrices
    bias_init: initializer for bias vectors
    positional_embedder:
  """
  yaml_tag = u'!SAAMSeqTransducer'

  @serializable_init
  @events.register_xnmt_handler
  def __init__(self, input_dim:int=512, layers:int=1, hidden_dim:int=512, head_count:int=8, ff_hidden_dim:int=2048,
               dropout:numbers.Real=Ref("exp_global.dropout", default=0.0), downsample_factor:int=1, diagonal_mask_width:int=None,
               ignore_masks:bool=False, plot_attention:typing.Optional[str]=None, nonlinearity:str="rectify", pos_encoding_type:typing.Optional[str]=None,
               pos_encoding_combine:str="concat", pos_encoding_size:int=40, max_len:int=1500, diag_gauss_mask:typing.Union[bool,numbers.Real]=False,
               square_mask_std:numbers.Real=True, cross_pos_encoding_type:typing.Optional[str]=None, ff_lstm:bool=False, kq_pos_encoding_type:typing.Optional[str]=None,
               kq_pos_encoding_size:int=40,
               param_init:xnmt.param_initializers.ParamInitializer=Ref("exp_global.param_init", default=bare(xnmt.param_initializers.GlorotInitializer)),
               bias_init:xnmt.param_initializers.ParamInitializer=Ref("exp_global.bias_init", default=bare(xnmt.param_initializers.ZeroInitializer)),
               positional_embedder=None, modules=None):
    self.input_dim = input_dim = (
            input_dim + (pos_encoding_size if (pos_encoding_type and pos_encoding_combine == "concat") else 0))
    self.hidden_dim = hidden_dim
    self.dropout = dropout
    self.layers = layers
    self.pos_encoding_type = pos_encoding_type
    self.pos_encoding_combine = pos_encoding_combine
    self.pos_encoding_size = pos_encoding_size
    self.max_len = max_len
    self.position_encoding_block = None
    if self.pos_encoding_type == "embedding":
      self.positional_embedder = \
        self.add_serializable_component("positional_embedder",
                                        positional_embedder,
                                        lambda: embedders.PositionEmbedder(max_pos=self.max_len,
                                                                           emb_dim=input_dim if self.pos_encoding_combine == "add" else self.pos_encoding_size))

    self.modules = self.add_serializable_component("modules", modules,
                                                   lambda: self.make_modules(layers=layers,
                                                                             plot_attention=plot_attention,
                                                                             hidden_dim=hidden_dim,
                                                                             downsample_factor=downsample_factor,
                                                                             input_dim=input_dim,
                                                                             head_count=head_count,
                                                                             ff_hidden_dim=ff_hidden_dim,
                                                                             diagonal_mask_width=diagonal_mask_width,
                                                                             ignore_masks=ignore_masks,
                                                                             nonlinearity=nonlinearity,
                                                                             diag_gauss_mask=diag_gauss_mask,
                                                                             square_mask_std=square_mask_std,
                                                                             cross_pos_encoding_type=cross_pos_encoding_type,
                                                                             ff_lstm=ff_lstm,
                                                                             kq_pos_encoding_type=kq_pos_encoding_type,
                                                                             kq_pos_encoding_size=kq_pos_encoding_size,
                                                                             dropout=dropout,
                                                                             param_init=param_init,
                                                                             bias_init=bias_init))

  def make_modules(self, layers, plot_attention, hidden_dim, downsample_factor, input_dim, head_count, ff_hidden_dim, diagonal_mask_width, ignore_masks, nonlinearity, diag_gauss_mask,
                   square_mask_std,cross_pos_encoding_type,ff_lstm,kq_pos_encoding_type,kq_pos_encoding_size,dropout,param_init,bias_init):
    modules = []
    for layer_i in range(layers):
      if plot_attention is not None:
        plot_attention_layer = f"{plot_attention}.layer_{layer_i}"
      else:
        plot_attention_layer = None
      modules.append(TransformerEncoderLayer(hidden_dim,
                                                  downsample_factor=downsample_factor,
                                                  input_dim=input_dim if layer_i == 0 else hidden_dim,
                                                  head_count=head_count, ff_hidden_dim=ff_hidden_dim,
                                                  diagonal_mask_width=diagonal_mask_width,
                                                  ignore_masks=ignore_masks,
                                                  plot_attention=plot_attention_layer,
                                                  nonlinearity=nonlinearity,
                                                  diag_gauss_mask=diag_gauss_mask,
                                                  square_mask_std=square_mask_std,
                                                  cross_pos_encoding_type=cross_pos_encoding_type,
                                                  ff_lstm=ff_lstm,
                                                  max_len=self.max_len,
                                                  kq_pos_encoding_type=kq_pos_encoding_type,
                                                  kq_pos_encoding_size=kq_pos_encoding_size,
                                                  dropout=dropout,
                                                  param_init=param_init[layer_i] if isinstance(param_init,
                                                                                               Sequence) else param_init,
                                                  bias_init=bias_init[layer_i] if isinstance(bias_init,
                                                                                             Sequence) else bias_init,
                                                  desc=f"layer_{layer_i}"))
    return modules

  def transduce(self, sent: ExpressionSequence) -> ExpressionSequence:
    if self.pos_encoding_type == "trigonometric":
      if self.position_encoding_block is None or self.position_encoding_block.shape[2] < len(sent):
        self.initialize_position_encoding(int(len(sent) * 1.2),
                                          self.input_dim if self.pos_encoding_combine == "add" else self.pos_encoding_size)
      encoding = dy.inputTensor(self.position_encoding_block[0, :, :len(sent)])
    elif self.pos_encoding_type == "embedding":
      encoding = self.positional_embedder.embed_sent(len(sent)).as_tensor()
    if self.pos_encoding_type:
      if self.pos_encoding_combine == "add":
        sent = ExpressionSequence(expr_tensor=sent.as_tensor() + encoding, mask=sent.mask)
      else:  # concat
        sent = ExpressionSequence(expr_tensor=dy.concatenate([sent.as_tensor(), encoding]),
                                                      mask=sent.mask)

    elif self.pos_encoding_type:
      raise ValueError(f"unknown encoding type {self.pos_encoding_type}")
    for module in self.modules:
      enc_sent = module.transduce(sent)
      sent = enc_sent
    self._final_states = [transducers.FinalTransducerState(sent[-1])]
    return sent

  def get_final_states(self):
    return self._final_states

  @events.handle_xnmt_event
  def on_set_train(self, val):
    for module in self.modules:
      module.set_dropout(self.dropout if val else 0.0)

  def initialize_position_encoding(self, length, n_units):
    # Implementation in the Google tensor2tensor repo
    channels = n_units
    position = np.arange(length, dtype='f')
    num_timescales = channels // 2
    log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
    inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.reshape(signal, [1, length, channels])
    self.position_encoding_block = np.transpose(signal, (0, 2, 1))
