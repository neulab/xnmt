import dynet as dy
import numpy as np
import collections
import itertools
import os
from typing import Union

# Reporting purposes
from lxml import etree
from xnmt.settings import settings

from xnmt.attender import MlpAttender
from xnmt.batcher import Batch, mark_as_batch, is_batched
from xnmt.decoder import MlpSoftmaxDecoder
from xnmt.embedder import SimpleWordEmbedder
from xnmt.events import register_xnmt_event_assign, handle_xnmt_event, register_xnmt_handler
from xnmt.model_base import GeneratorModel, EventTrigger
from xnmt.inference import SimpleInference
from xnmt.input import Input, SimpleSentenceInput
import xnmt.length_normalization
from xnmt.loss import FactoredLossExpr
from xnmt.loss_calculator import LossCalculator
from xnmt.lstm import BiLSTMSeqTransducer
from xnmt.output import TextOutput
import xnmt.plot
from xnmt.reports import Reportable
from xnmt.persistence import serializable_init, Serializable, bare
from xnmt.search_strategy import BeamSearch
from collections import namedtuple
from xnmt.vocab import Vocab
from xnmt.constants import EPSILON

TranslatorOutput = namedtuple('TranslatorOutput', ['state', 'logsoftmax', 'attention'])

class Translator(GeneratorModel):
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  def calc_loss(self, src: Union[Batch, Input], trg: Union[Batch, Input],
                loss_calculator: LossCalculator) -> FactoredLossExpr:
    '''Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.
    
    Args:
      src: The source, a sentence or a batch of sentences.
      trg: The target, a sentence or a batch of sentences.
      loss_calculator: loss calculator.
    
    Returns:
      A (possibly batched) expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Translator subclasses')

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs. If not specified, word IDs are generated instead.

    Args:
      trg_vocab (Vocab): target vocab, or None to generate word IDs
    """
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

  def get_primary_loss(self):
    return "mle"

  def output_one_step(self):
    raise NotImplementedError()

  def get_nobp_state(self, state):
    output_state = state.rnn_state.output()
    if type(output_state) == EnsembleListDelegate:
      for i in range(len(output_state)):
        output_state[i] = dy.nobackprop(output_state[i])
    else:
      output_state = dy.nobackprop(output_state)
    return output_state

class DefaultTranslator(Translator, Serializable, Reportable, EventTrigger):
  '''
  A default translator based on attentional sequence-to-sequence models.

  Args:
    src_reader (InputReader): A reader for the source side.
    trg_reader (InputReader): A reader for the target side.
    src_embedder (Embedder): A word embedder for the input language
    encoder (Transducer): An encoder to generate encoded inputs
    attender (Attender): An attention module
    trg_embedder (Embedder): A word embedder for the output language
    decoder (Decoder): A decoder
    inference (SimpleInference): The default inference strategy used for this model
    calc_global_fertility (bool):
    calc_attention_entropy (bool):
  '''

  yaml_tag = '!DefaultTranslator'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, src_reader, trg_reader, src_embedder=bare(SimpleWordEmbedder),
               encoder=bare(BiLSTMSeqTransducer), attender=bare(MlpAttender),
               trg_embedder=bare(SimpleWordEmbedder), decoder=bare(MlpSoftmaxDecoder),
               inference=bare(SimpleInference), search_strategy=bare(BeamSearch),
               calc_global_fertility=False, calc_attention_entropy=False):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.calc_global_fertility = calc_global_fertility
    self.calc_attention_entropy = calc_attention_entropy
    self.inference = inference
    self.search_strategy = search_strategy

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".attender.input_dim", ".decoder.input_dim"},
            {".attender.state_dim", ".decoder.rnn_layer.hidden_dim"},
            {".trg_embedder.emb_dim", ".decoder.trg_embed_dim"}]

  def initialize_generator(self, **kwargs):
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def calc_loss(self, src, trg, loss_calculator):
    self.start_sent(src)
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder(embeddings)
    self.attender.init_sent(encodings)
    # Initialize the hidden state from the encoder
    ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
    initial_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
    # Compose losses
    model_loss = FactoredLossExpr()
    model_loss.add_factored_loss_expr(loss_calculator(self, initial_state, src, trg))

    if self.calc_global_fertility or self.calc_attention_entropy:
      # philip30: I assume that attention_vecs is already masked src wisely.
      # Now applying the mask to the target
      masked_attn = self.attender.attention_vecs
      if trg.mask is not None:
        trg_mask = trg.mask.get_active_one_mask().transpose()
        masked_attn = [dy.cmult(attn, dy.inputTensor(mask, batched=True)) for attn, mask in zip(masked_attn, trg_mask)]

    if self.calc_global_fertility:
      model_loss.add_loss("fertility", self.global_fertility(masked_attn))
    if self.calc_attention_entropy:
      model_loss.add_loss("H(attn)", self.attention_entropy(masked_attn))

    return model_loss

  def generate(self, src, idx, search_strategy, src_mask=None, forced_trg_ids=None):
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    # Generating outputs
    outputs = []
    for sents in src:
      self.start_sent(src)
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder(embeddings)
      self.attender.init_sent(encodings)
      ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
      initial_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
      search_outputs = search_strategy.generate_output(self, initial_state,
                                                       src_length=[len(sents)],
                                                       forced_trg_ids=forced_trg_ids)
      best_output = sorted(search_outputs, key=lambda x: x.score[0], reverse=True)[0]
      output_actions = [x for x in best_output.word_ids[0]]
      attentions = [x for x in best_output.attentions[0]]
      score = best_output.score[0]
      # In case of reporting
      if self.report_path is not None:
        if self.reporting_src_vocab:
          src_words = [self.reporting_src_vocab[w] for w in sents]
        else:
          src_words = ['' for w in sents]
        trg_words = [self.trg_vocab[w] for w in output_actions]
        # Attentions
        attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
        # Segmentation
        segment = self.get_report_resource("segmentation")
        if segment is not None:
          segment = [int(x[0]) for x in segment]
          src_inp = [x[0] for x in self.encoder.apply_segmentation(src_words, segment)]
        else:
          src_inp = src_words
        # Other Resources
        self.set_report_input(idx, src_inp, trg_words, attentions)
        self.set_report_resource("src_words", src_words)
        self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
        self.generate_report(self.report_type)
      # Append output to the outputs
      outputs.append(TextOutput(actions=output_actions,
                                vocab=self.trg_vocab if hasattr(self, "trg_vocab") else None,
                                score=score))
    self.outputs = outputs
    return outputs

  def global_fertility(self, a):
    return dy.sum_elems(dy.square(1 - dy.esum(a)))

  def attention_entropy(self, a):
    entropy = []
    for a_i in a:
      a_i += EPSILON
      entropy.append(dy.cmult(a_i, dy.log(a_i)))

    return -dy.sum_elems(dy.esum(entropy))

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    
    Args:
      src_vocab (Vocab):
    """
    self.reporting_src_vocab = src_vocab

  def output_one_step(self, current_word, current_state):
    if current_word is not None:
      if type(current_word) == int:
        current_word = [current_word]
      if type(current_word) == list or type(current_word) == np.ndarray:
        current_word = xnmt.batcher.mark_as_batch(current_word)
      current_word_embed = self.trg_embedder.embed(current_word)
      next_state = self.decoder.add_input(current_state, current_word_embed)
    else:
      next_state = current_state
    next_state.context = self.attender.calc_context(next_state.rnn_state.output())
    next_logsoftmax = self.decoder.get_scores_logsoftmax(next_state)
    return TranslatorOutput(next_state, next_logsoftmax, self.attender.get_last_attention())

  @register_xnmt_event_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.get_report_input()
    path_to_report = self.get_report_path()
    html = etree.Element('html')
    head = etree.SubElement(html, 'head')
    title = etree.SubElement(head, 'title')
    body = etree.SubElement(html, 'body')
    report = etree.SubElement(body, 'h1')
    if idx is not None:
      title.text = report.text = 'Translation Report for Sentence %d' % (idx)
    else:
      title.text = report.text = 'Translation Report'
    main_content = etree.SubElement(body, 'div', name='main_content')

    # Generating main content
    captions = ["Source Words", "Target Words"]
    inputs = [src, trg]
    for caption, inp in zip(captions, inputs):
      if inp is None: continue
      sent = ' '.join(inp)
      p = etree.SubElement(main_content, 'p')
      p.text = f"{caption}: {sent}"

    # Generating attention
    if not any([src is None, trg is None, att is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      attention_file = f"{path_to_report}.attention.png"
      att_img = etree.SubElement(attention, 'img')
      att_img_src = f"{path_to_report}.attention.png"
      att_img.attrib['src'] = os.path.basename(att_img_src)
      att_img.attrib['alt'] = 'attention matrix'
      xnmt.plot.plot_attention(src, trg, att, file_name = attention_file)

    # return the parent context to be used as child context
    return html

  @handle_xnmt_event
  def on_file_report(self):
    idx, src, trg, attn = self.get_report_input()
    assert attn.shape == (len(src), len(trg))
    col_length = []
    for word in trg:
      col_length.append(max(len(word), 6))
    col_length.append(max(len(x) for x in src))
    with open(self.get_report_path() + ".attention.txt", encoding='utf-8', mode='w') as attn_file:
      for i in range(len(src)+1):
        if i == 0:
          words = trg + [""]
        else:
          words = ["%.4f" % (f) for f in attn[i-1]] + [src[i-1]]
        str_format = ""
        for length in col_length:
          str_format += "{:%ds}" % (length+2)
        print(str_format.format(*words), file=attn_file)

  
class TransformerTranslator(Translator, Serializable, Reportable, EventTrigger):
  '''
  A translator based on the transformer model.

  Args:
    src_reader (InputReader): A reader for the source side.
    src_embedder (Embedder): A word embedder for the input language
    encoder (TransformerEncoder): An encoder to generate encoded inputs
    trg_reader (InputReader): A reader for the target side.
    trg_embedder (Embedder): A word embedder for the output language
    decoder (TransformerDecoder): A decoder
    inference (SimpleInference): The default inference strategy used for this model
    input_dim (int):
  '''

  yaml_tag = '!TransformerTranslator'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, src_reader, src_embedder, encoder, trg_reader, trg_embedder, decoder, inference=None, input_dim=512):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.input_dim = input_dim
    self.inference = inference
    self.scale_emb = self.input_dim ** 0.5
    self.max_input_len = 500
    self.initialize_position_encoding(self.max_input_len, input_dim)  # TODO: parametrize this

  def initialize_generator(self, **kwargs):
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def initialize_training_strategy(self, training_strategy):
    self.loss_calculator = training_strategy

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab

  def make_attention_mask(self, source_block, target_block):
    mask = (target_block[:, None, :] <= 0) * (source_block[:, :, None] <= 0)
    # (batch, source_length, target_length)
    return mask

  def make_history_mask(self, block):
    batch, length = block.shape
    arange = np.arange(length)
    history_mask = (arange[None,] <= arange[:, None])[None,]
    history_mask = np.broadcast_to(history_mask, (batch, length, length))
    return history_mask

  def mask_embeddings(self, embeddings, mask):
    """
    We convert the embeddings of masked input sequence to zero vector
    """
    (embed_dim, _), _ = embeddings.dim()
    temp_mask = np.repeat(1. - mask[:, None, :], embed_dim, axis=1)
    temp_mask = dy.inputTensor(np.moveaxis(temp_mask, [1, 0, 2], [0, 2, 1]), batched=True)
    embeddings = dy.cmult(embeddings, temp_mask)
    return embeddings

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

  def make_input_embedding(self, emb_block, length):
    if length > self.max_input_len:
      self.initialize_position_encoding(2 * length, self.input_dim)
      self.max_input_len = 2 * length
    emb_block = emb_block * self.scale_emb
    emb_block += dy.inputTensor(self.position_encoding_block[0, :, :length])
    return emb_block

  def sentence_block_embed(self, embed, x, mask):
    batch, length = x.shape
    x_mask = mask.reshape((batch * length,))
    _, units = embed.shape()  # According to updated Dynet
    e = dy.concatenate_cols([dy.zeros(units) if x_mask[j] == 1 else dy.lookup(embed, id_) for j, id_ in enumerate(x.reshape((batch * length,)))])
    e = dy.reshape(e, (units, length), batch_size=batch)
    return e

  def calc_loss(self, src, trg, loss_cal=None, infer_prediction=False):
    self.start_sent(src)
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    if not xnmt.batcher.is_batched(trg):
      trg = xnmt.batcher.mark_as_batch([trg])
    src_words = np.array([[Vocab.SS] + x.words for x in src])
    batch_size, src_len = src_words.shape

    if isinstance(src.mask, type(None)):
      src_mask = np.zeros((batch_size, src_len), dtype=np.int)
    else:
      src_mask = np.concatenate([np.zeros((batch_size, 1), dtype=np.int), src.mask.np_arr.astype(np.int)], axis=1)

    src_embeddings = self.sentence_block_embed(self.src_embedder.embeddings, src_words, src_mask)
    src_embeddings = self.make_input_embedding(src_embeddings, src_len)

    trg_words = np.array(list(map(lambda x: [Vocab.SS] + x.words[:-1], trg)))
    batch_size, trg_len = trg_words.shape

    if isinstance(trg.mask, type(None)):
      trg_mask = np.zeros((batch_size, trg_len), dtype=np.int)
    else:
      trg_mask = trg.mask.np_arr.astype(np.int)

    trg_embeddings = self.sentence_block_embed(self.trg_embedder.embeddings, trg_words, trg_mask)
    trg_embeddings = self.make_input_embedding(trg_embeddings, trg_len)

    xx_mask = self.make_attention_mask(src_mask, src_mask)
    xy_mask = self.make_attention_mask(trg_mask, src_mask)
    yy_mask = self.make_attention_mask(trg_mask, trg_mask)
    yy_mask *= self.make_history_mask(trg_mask)

    z_blocks = self.encoder(src_embeddings, xx_mask)
    h_block = self.decoder(trg_embeddings, z_blocks, xy_mask, yy_mask)

    if infer_prediction:
      y_len = h_block.dim()[0][1]
      last_col = dy.pick(h_block, dim=1, index=y_len - 1)
      logits = self.decoder.output(last_col)
      return logits

    ref_list = list(itertools.chain.from_iterable(map(lambda x: x.words, trg)))
    concat_t_block = (1 - trg_mask.ravel()).reshape(-1) * np.array(ref_list)
    loss = self.decoder.output_and_loss(h_block, concat_t_block)
    return FactoredLossExpr({"mle": loss})

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None, search_strategy=None):
    self.start_sent(src)
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []

    trg = SimpleSentenceInput([0])

    if not xnmt.batcher.is_batched(trg):
      trg = xnmt.batcher.mark_as_batch([trg])

    output_actions = []
    score = 0.

    # TODO Fix this with output_one_step and use the appropriate search_strategy
    self.max_len = 100 # This is a temporary hack
    for _ in range(self.max_len):
      dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
      log_prob_tail = self.calc_loss(src, trg, loss_cal=None, infer_prediction=True)
      ys = np.argmax(log_prob_tail.npvalue(), axis=0).astype('i')
      if ys == Vocab.ES:
        output_actions.append(ys)
        break
      output_actions.append(ys)
      trg = SimpleSentenceInput(output_actions + [0])
      if not xnmt.batcher.is_batched(trg):
        trg = xnmt.batcher.mark_as_batch([trg])

    # In case of reporting
    sents = src[0]
    if self.report_path is not None:
      src_words = [self.reporting_src_vocab[w] for w in sents]
      trg_words = [self.trg_vocab[w] for w in output_actions]
      self.set_report_input(idx, src_words, trg_words)
      self.set_report_resource("src_words", src_words)
      self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
      self.generate_report(self.report_type)

    # Append output to the outputs
    if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
      outputs.append(TextOutput(output_actions, self.trg_vocab))
    else:
      outputs.append((output_actions, score))

    return outputs

class EnsembleTranslator(Translator, Serializable, EventTrigger):
  '''
  A translator that decodes from an ensemble of DefaultTranslator models.

  Args:
    models: A list of DefaultTranslator instances; for all models, their
      src_reader.vocab and trg_reader.vocab has to match (i.e., provide
      identical conversions to) those supplied to this class.
    src_reader (InputReader): A reader for the source side.
    trg_reader (InputReader): A reader for the target side.
    inference (SimpleInference): The inference strategy used for this ensemble.
  '''

  yaml_tag = '!EnsembleTranslator'

  @register_xnmt_handler
  @serializable_init
  def __init__(self, models, src_reader, trg_reader, inference=bare(SimpleInference)):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.models = models
    self.inference = inference

    # perform checks to verify the models can logically be ensembled
    for i, model in enumerate(self.models):
      if hasattr(self.src_reader, "vocab") or hasattr(model.src_reader, "vocab"):
        assert self.src_reader.vocab.is_compatible(model.src_reader.vocab), \
          f"src_reader.vocab is not compatible with model {i}"
      assert self.trg_reader.vocab.is_compatible(model.trg_reader.vocab), \
        f"trg_reader.vocab is not compatible with model {i}"

    # proxy object used for generation, to avoid code duplication
    self._proxy = DefaultTranslator(
      self.src_reader,
      self.trg_reader,
      EnsembleListDelegate([model.src_embedder for model in self.models]),
      EnsembleListDelegate([model.encoder for model in self.models]),
      EnsembleListDelegate([model.attender for model in self.models]),
      EnsembleListDelegate([model.trg_embedder for model in self.models]),
      EnsembleDecoder([model.decoder for model in self.models])
    )

  def shared_params(self):
    shared = [params for model in self.models for params in model.shared_params()]
    return shared

  def set_trg_vocab(self, trg_vocab=None):
    self._proxy.set_trg_vocab(trg_vocab=trg_vocab)

  def initialize_generator(self, **kwargs):
    self._proxy.initialize_generator(**kwargs)

  def calc_loss(self, src, trg, loss_calculator):
    sub_losses = collections.defaultdict(list)
    for model in self.models:
      for loss_name, loss in model.calc_loss(src, trg, loss_calculator).loss_values.items():
        sub_losses[loss_name].append(loss)
    model_loss = FactoredLossExpr()
    for loss_name, losslist in sub_losses.items():
      # TODO: dy.average(losslist)  _or_  dy.esum(losslist) / len(self.models) ?
      #       -- might not be the same if not all models return all losses
      model_loss.add_loss(loss_name, dy.average(losslist))
    return model_loss

  def generate(self, src, idx, search_strategy, src_mask=None, forced_trg_ids=None):
    return self._proxy.generate(src, idx, search_strategy, src_mask=src_mask, forced_trg_ids=forced_trg_ids)

class EnsembleListDelegate(object):
  '''
  Auxiliary object to wrap a list of objects for ensembling.

  This class can wrap a list of objects that exist in parallel and do not need
  to interact with each other. The main functions of this class are:

  - All attribute access and function calls are delegated to the wrapped objects.
  - When wrapped objects return values, the list of all returned values is also
    wrapped in an EnsembleListDelegate object.
  - When EnsembleListDelegate objects are supplied as arguments, they are
    "unwrapped" so the i-th object receives the i-th element of the
    EnsembleListDelegate argument.
  '''

  def __init__(self, objects):
    assert isinstance(objects, (tuple, list))
    self._objects = objects

  def __getitem__(self, key):
    return self._objects[key]

  def __setitem__(self, key, value):
    self._objects[key] = value

  def __iter__(self):
    return self._objects.__iter__()

  def __call__(self, *args, **kwargs):
    return self.__getattr__('__call__')(*args, **kwargs)

  def __len__(self):
    return len(self._objects)

  def __getattr__(self, attr):
    def unwrap(list_idx, args, kwargs):
      args = [arg if not isinstance(arg, EnsembleListDelegate) else arg[list_idx] \
              for arg in args]
      kwargs = {key: val if not isinstance(val, EnsembleListDelegate) else val[list_idx] \
                for key, val in kwargs.items()}
      return args, kwargs

    attrs = [getattr(obj, attr) for obj in self._objects]
    if callable(attrs[0]):
      def wrapper_func(*args, **kwargs):
        ret = []
        for i, attr_ in enumerate(attrs):
          args_i, kwargs_i = unwrap(i, args, kwargs)
          ret.append(attr_(*args_i, **kwargs_i))
        if all(val is None for val in ret):
          return None
        else:
          return EnsembleListDelegate(ret)
      return wrapper_func
    else:
      return EnsembleListDelegate(attrs)

  def __setattr__(self, attr, value):
    if not attr.startswith('_'):
      if isinstance(value, EnsembleListDelegate):
        for i, obj in enumerate(self._objects):
          setattr(obj, attr, value[i])
      else:
        for obj in self._objects:
          setattr(obj, attr, value)
    else:
      self.__dict__[attr] = value

  def __repr__(self):
    return "EnsembleListDelegate([" + ', '.join(repr(elem) for elem in self._objects) + "])"


class EnsembleDecoder(EnsembleListDelegate):
  '''
  Auxiliary object to wrap a list of decoders for ensembling.

  This behaves like an EnsembleListDelegate, except that it overrides
  get_scores() to combine the individual decoder's scores.

  Currently only supports averaging.
  '''
  def get_scores_logsoftmax(self, mlp_dec_states):
    scores = [obj.get_scores_logsoftmax(dec_state) for obj, dec_state in zip(self._objects, mlp_dec_states)]
    return dy.average(scores)

