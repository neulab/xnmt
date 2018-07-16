import dynet as dy
import numpy as np

from xnmt import attender, batcher, embedder, events, inference, input_reader, loss, lstm, model_base, output, \
  reports, scorer, transducer, transform, vocab
from xnmt.persistence import serializable_init, Serializable, bare

class SeqLabeler(model_base.ConditionedModel, model_base.GeneratorModel, Serializable, reports.Reportable,
                 model_base.EventTrigger):
  """
  A simple sequence labeler based on an encoder and an output softmax layer.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    transform: A transform to be applied before making predictions
    scorer: The class to actually make predictions
    inference: The inference method used for this model
    auto_cut_pad: If ``True``, cut or pad target sequences so the match the length of the encoded inputs.
                  If ``False``, an error is thrown if there is a length mismatch.
  """

  yaml_tag = '!SeqLabeler'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader:input_reader.InputReader,
               trg_reader:input_reader.InputReader,
               src_embedder:embedder.Embedder=bare(embedder.SimpleWordEmbedder),
               encoder:transducer.SeqTransducer=bare(lstm.BiLSTMSeqTransducer),
               transform:transform.Transform=bare(transform.NonLinear),
               scorer:scorer.Scorer=bare(scorer.Softmax),
               inference:inference.Inference=bare(inference.IndependentOutputInference),
               auto_cut_pad:bool=False):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.transform = transform
    self.scorer = scorer
    self.inference = inference
    self.auto_cut_pad = auto_cut_pad

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},]

  def get_primary_loss(self):
    return "mle"

  def _encode_src(self, src):
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    encodings_tensor = encodings.as_tensor()
    ((hidden_dim, seq_len), batch_size) = encodings.dim()
    encoding_reshaped = dy.reshape(encodings_tensor, (hidden_dim,), batch_size=batch_size * seq_len)
    outputs = self.transform(encoding_reshaped)
    return batch_size, encodings, outputs, seq_len

  def calc_loss(self, src, trg, loss_calculator):
    assert batcher.is_batched(src) and batcher.is_batched(trg)
    batch_size, encodings, outputs, seq_len = self._encode_src(src)

    if trg.sent_len() != seq_len:
      if self.auto_cut_pad:
        trg = self._cut_or_pad_targets(seq_len, trg)
      else:
        raise ValueError(f"src/trg length do not match: {seq_len} != {len(trg[0])}")

    ref_action = np.asarray([sent.words for sent in trg]).reshape((seq_len * batch_size,))
    loss_expr_perstep = self.scorer.calc_loss(outputs, batcher.mark_as_batch(ref_action))
    # loss_expr_perstep = dy.pickneglogsoftmax_batch(outputs, ref_action)
    loss_expr_perstep = dy.reshape(loss_expr_perstep, (seq_len,), batch_size=batch_size)
    if trg.mask:
      loss_expr_perstep = dy.cmult(loss_expr_perstep, dy.inputTensor(1.0-trg.mask.np_arr.T, batched=True))
    loss_expr = dy.sum_elems(loss_expr_perstep)

    model_loss = loss.FactoredLossExpr()
    model_loss.add_loss("mle", loss_expr)

    return model_loss

  def _cut_or_pad_targets(self, seq_len, trg):
    old_mask = trg.mask
    if len(trg[0]) > seq_len:
      trunc_len = len(trg[0]) - seq_len
      trg = batcher.mark_as_batch([trg_sent.get_truncated_sent(trunc_len=trunc_len) for trg_sent in trg])
      if old_mask:
        trg.mask = batcher.Mask(np_arr=old_mask.np_arr[:, :-trunc_len])
    else:
      pad_len = seq_len - len(trg[0])
      trg = batcher.mark_as_batch([trg_sent.get_padded_sent(token=vocab.Vocab.ES, pad_len=pad_len) for trg_sent in trg])
      if old_mask:
        trg.mask = np.pad(old_mask.np_arr, pad_width=((0, 0), (0, pad_len)), mode="constant", constant_values=1)
    return trg

  def generate(self, src, idx, forced_trg_ids=None, normalize_scores = False):
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
      if forced_trg_ids:
        forced_trg_ids = batcher.mark_as_batch([forced_trg_ids])
    assert src.batch_size() == 1, "batch size > 1 not properly tested"

    batch_size, encodings, outputs, seq_len = self._encode_src(src)
    score_expr = self.scorer.calc_log_softmax(outputs) if normalize_scores else self.scorer.calc_scores(outputs)
    scores = score_expr.npvalue() # vocab_size x seq_len

    if forced_trg_ids:
      output_actions = forced_trg_ids
    else:
      output_actions = [np.argmax(scores[:, j]) for j in range(seq_len)]
    score = np.sum([scores[output_actions[j], j] for j in range(seq_len)])

    outputs = [output.TextOutput(actions=output_actions,
                      vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                      score=score)]

    return outputs

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab (vocab.Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab
