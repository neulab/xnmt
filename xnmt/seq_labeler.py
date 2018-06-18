import dynet as dy
import numpy as np

from xnmt import attender, batcher, embedder, events, inference, input_reader, loss, lstm,  mlp, model_base, output, \
  reports, transducer, vocab
from xnmt.persistence import serializable_init, Serializable, bare

class SeqLabeler(model_base.GeneratorModel, Serializable, reports.Reportable, model_base.EventTrigger):
  """
  A default translator based on attentional sequence-to-sequence models.

  Args:
    src_reader (InputReader): A reader for the source side.
    trg_reader (InputReader): A reader for the target side.
    src_embedder (Embedder): A word embedder for the input language
    encoder (Transducer): An encoder to generate encoded inputs
    decoder (MLP): final prediction layer
    inference (inference.SequenceInference): The default inference method used for this model
  """

  yaml_tag = '!SeqLabeler'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader:input_reader.InputReader,
               trg_reader:input_reader.InputReader,
               src_embedder:embedder.Embedder=bare(embedder.SimpleWordEmbedder),
               encoder:transducer.SeqTransducer=bare(lstm.BiLSTMSeqTransducer),
               decoder:mlp.MLP=bare(mlp.MLP),
               inference=bare(inference.SequenceInference),
               auto_cut_pad:bool=False):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder
    self.inference = inference
    self.auto_cut_pad = auto_cut_pad

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},]

  def get_primary_loss(self):
    return "mle"

  def calc_loss(self, src, trg, loss_calculator):
    assert batcher.is_batched(src) and batcher.is_batched(trg)
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
    ((hidden_dim, seq_len), batch_size) = encodings.dim()
    if len(trg[0]) != seq_len:
      if self.auto_cut_pad:
        old_mask = trg.mask
        if len(trg[0]) > seq_len:
          trunc_len = len(trg[0])-seq_len
          trg = batcher.mark_as_batch([trg_sent.get_truncated_sent(trunc_len=trunc_len) for trg_sent in trg])
          if old_mask:
            trg.mask = batcher.Mask(np_arr=old_mask.np_arr[:,:-trunc_len])
        else:
          pad_len = seq_len-len(trg[0])
          trg = batcher.mark_as_batch([trg_sent.get_padded_sent(token=vocab.Vocab.ES, pad_len=pad_len) for trg_sent in trg])
          if old_mask:
            trg.mask = np.pad(old_mask.np_arr, pad_width=((0,0), (0,pad_len)), mode="constant", constant_values=1)
      else:
        raise ValueError(f"src/trg length do not match: {seq_len} != {len(trg[0])}")
    encodings_tensor = encodings.as_tensor()
    encoding_reshaped = dy.reshape(encodings_tensor, (hidden_dim,), batch_size=batch_size*seq_len)
    outputs = self.decoder(encoding_reshaped)
    masked_outputs = dy.cmult(outputs, dy.inputTensor(1.0 - encodings.mask.np_arr.reshape((seq_len * batch_size,)),
                                                      batched=True))
    ref_action = np.asarray([sent.words for sent in trg]).reshape((seq_len * batch_size,))
    loss_expr_perstep = dy.pickneglogsoftmax_batch(masked_outputs, ref_action)
    loss_expr = dy.sum_elems(dy.reshape(loss_expr_perstep, (seq_len,), batch_size=batch_size))

    model_loss = loss.FactoredLossExpr()
    model_loss.add_loss("mle", loss_expr)

    return model_loss

  def generate(self, src, idx, search_strategy, src_mask=None, forced_trg_ids=None):
    if not batcher.is_batched(src):
      src = batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    assert len(src) == 1, "batch size > 1 not properly tested"

    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
    encodings_tensor = encodings.as_tensor()
    ((hidden_dim, seq_len), batch_size) = encodings_tensor.dim()
    encoding_reshaped = dy.reshape(encodings_tensor, (hidden_dim,), batch_size=batch_size*seq_len)
    outputs = self.decoder(encoding_reshaped)
    loss_expr_perstep = dy.log_softmax(outputs)
    scores = loss_expr_perstep.npvalue() # vocab_size x seq_len
    output_actions = [np.argmax(scores[:,j]) for j in range(seq_len)]
    score = np.sum([np.max(scores[:,j]) for j in range(seq_len)])

    outputs = [output.TextOutput(actions=output_actions,
                      vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                      score=score)]

    self.outputs = outputs
    return outputs

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab (vocab.Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor
