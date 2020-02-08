from typing import List
import numbers

import xnmt
import xnmt.tensor_tools as tt
from xnmt import events, expression_seqs, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.persistence import serializable_init, Serializable, bare, Ref

if xnmt.backend_dynet:
  import dynet as dy
if xnmt.backend_torch:
  import torch
  import torch.nn as nn

# Note: alternatively, this could wrap "PositionEmbedder", but it seems to me
#       that PositionEmbedder is probably not necessary in the first place, so
#       it probably makes more sense to have this as a SeqTransducer that
#       adds positional embeddings to an input
@xnmt.require_dynet
class PositionalSeqTransducerDynet(transducers.SeqTransducer, Serializable):
  yaml_tag = '!PositionalSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               max_pos: numbers.Integral,
               op: str = 'sum',
               emb_type: str = 'param',
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))) \
          -> None:
    """
    max_pos: largest embedded position
    op: how to combine positional encodings with the original encodings, can be "sum" or "concat"
    type: what type of embddings to use, "param"=parameterized (others, such as the trigonometric embeddings are todo)
    input_dim: embedding size
    dropout: apply dropout to output of this transducer
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.input_dim = input_dim
    self.dropout = dropout
    self.op = op
    self.emb_type = emb_type
    param_init = param_init
    dim = (self.input_dim, max_pos)
    my_params = param_collections.ParamManager.my_params(self)
    self.embedder = my_params.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  @ events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, src: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    sent_len = src.sent_len()
    embeddings = dy.strided_select(dy.parameter(self.embedder), [1,1], [0,0], [self.input_dim, sent_len])
    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = dy.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')
    if self.train and self.dropout > 0.0:
      output = dy.dropout(output, self.dropout)
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output, mask=src.mask)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq

@xnmt.require_torch
class PositionalSeqTransducerTorch(transducers.SeqTransducer, Serializable):
  yaml_tag = '!PositionalSeqTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               max_pos: numbers.Integral,
               op: str = 'sum',
               emb_type: str = 'param',
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))) \
          -> None:
    """
    max_pos: largest embedded position
    op: how to combine positional encodings with the original encodings, can be "sum" or "concat"
    type: what type of embddings to use, "param"=parameterized (others, such as the trigonometric embeddings are todo)
    input_dim: embedding size
    dropout: apply dropout to output of this transducer
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.input_dim = input_dim
    self.dropout = dropout
    self.op = op
    self.emb_type = emb_type
    my_params = param_collections.ParamManager.my_params(self)
    self.embeddings = nn.Embedding(self.max_pos, self.input_dim).to(xnmt.device)
    my_params.append(self.embeddings)
    my_params.init_params(param_init)

  @ events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, src: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    sent_len = src.sent_len()
    batch_size = tt.batch_size(src[0])
    embeddings = self.embeddings(torch.tensor([list(range(sent_len))] * batch_size).to(xnmt.device))
    # embeddings = dy.strided_select(dy.parameter(self.embedder), [1,1], [0,0], [self.input_dim, sent_len])
    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = tt.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')
    if self.train and self.dropout > 0.0:
      output = tt.dropout(output, self.dropout)
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output, mask=src.mask)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq

PositionalSeqTransducer = xnmt.resolve_backend(PositionalSeqTransducerDynet, PositionalSeqTransducerTorch)
