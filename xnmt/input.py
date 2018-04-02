import logging
logger = logging.getLogger('xnmt')

import ast
import io
import six
import re

import numpy as np
from xnmt.serialize.serializable import Serializable
from xnmt.vocab import Vocab, Rule, RuleVocab

###### Classes representing single inputs

class Input(object):
  """
  A template class to represent a single input of any type.
  """
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Input subclasses")

  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.
    
    Args:
      token: padding token
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.Input: padded sent
    """
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sent, represented as a list of tokens
  
  Args:
    words (List[int]): list of integer word ids
    vocab (Vocab):
  """
  def __init__(self, words, vocab=None):
    self.words = words
    self.vocab = vocab

  def __len__(self):
    return len(self.words)

  def __getitem__(self, key):
    return self.words[key]

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.
    
    Args:
      token (int): padding token
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.SimpleSentenceInput: padded sentence
    """
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([token] * pad_len)
    return self.__class__(new_words, self.vocab)

  def __str__(self):
    return " ".join(map(str, self.words))

class AnnotatedSentenceInput(SimpleSentenceInput):
  def __init__(self, words, vocab=None):
    super(AnnotatedSentenceInput, self).__init__(words, vocab)
    self.annotation = {}

  def annotate(self, key, value):
    self.annotation[key] = value

  def get_padded_sent(self, token, pad_len):
    sent = super(AnnotatedSentenceInput, self).get_padded_sent(token, pad_len)
    sent.annotation = self.annotation
    return sent

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens.
  
  Args:
    nparr: numpy array
  """
  def __init__(self, nparr):
    self.nparr = nparr

  def __len__(self):
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def __getitem__(self, key):
    return self.nparr.__getitem__(key)

  def get_padded_sent(self, token, pad_len):
    """
    Return padded version of the sent.
    
    Args:
      token: None (replicate last frame) or 0 (pad zeros)
      pad_len (int): number of tokens to append
    Returns:
      xnmt.input.ArrayInput: padded sentence
    """
    if pad_len == 0:
      return self
    if token is None:
      new_nparr = np.append(self.nparr, np.broadcast_to(np.reshape(self.nparr[:,-1], (self.nparr.shape[0], 1)), (self.nparr.shape[0], pad_len)), axis=1)
    elif token == 0:
      new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
    else:
      raise NotImplementedError(f"currently only support 'None' or '0' as, but got '{token}'")
    return ArrayInput(new_nparr)

  def get_array(self):
    return self.nparr

###### Classes that will read in a file and turn it into an input

class InputReader(object):
  def read_sents(self, filename, filter_ids=None):
    """
    :param filename: data file
    :param filter_ids: only read sentences with these ids (0-indexed)
    :returns: iterator over sentences from filename
    """
    raise RuntimeError("Input readers must implement the read_sents function")

  def count_sents(self, filename):
    """
    :param filename: data file
    :returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def freeze(self):
    pass

class BaseTextReader(InputReader):
  def count_sents(self, filename):
    f = io.open(filename, encoding='utf-8')
    try:
      return sum(1 for _ in f)
    finally:
      f.close()

  def iterate_filtered(self, filename, filter_ids=None):
    """
    :param filename: data file (text file)
    :param filter_ids:
    :returns: iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with io.open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield line
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files,
  with one sent per line.
  """
  yaml_tag = u'!PlainTextReader'
  def __init__(self, vocab=None, include_vocab_reference=False):
    self.vocab = vocab
    self.include_vocab_reference = include_vocab_reference
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    vocab_reference = self.vocab if self.include_vocab_reference else None
    return six.moves.map(lambda l: SimpleSentenceInput([self.vocab.convert(word) for word in l.strip().split()] + \
                                                       [self.vocab.convert(Vocab.ES_STR)], vocab_reference),
                         self.iterate_filtered(filename, filter_ids))

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param("vocab", self.vocab)

  def count_words(self, trg_words):
    trg_cnt = 0
    for x in trg_words:
      if type(x) == int:
        trg_cnt += 1 if x != Vocab.ES else 0
      else:
        trg_cnt += sum([1 if y != Vocab.ES else 0 for y in x])
    return trg_cnt

  def vocab_size(self):
    return len(self.vocab)

class SegmentationTextReader(PlainTextReader):
  yaml_tag = u'!SegmentationTextReader'

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    def convert(line, segmentation):
      line = line.strip().split()
      ret = AnnotatedSentenceInput(list(six.moves.map(self.vocab.convert, line)) + [self.vocab.convert(Vocab.ES_STR)])
      ret.annotate("segment", list(six.moves.map(int, segmentation.strip().split())))
      return ret

    if type(filename) != list:
      try:
        filename = ast.literal_eval(filename)
      except:
        logger.debug("Reading %s with a PlainTextReader instead..." % filename)
        return super(SegmentationTextReader, self).read_sents(filename)

    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    data = []
    with io.open(filename[0], encoding='utf-8') as char_inp,\
         io.open(filename[1], encoding='utf-8') as seg_inp:
      for sent_count, (char_line, seg_line) in enumerate(zip(char_inp, seg_inp)):
        if filter_ids is None or sent_count in filter_ids:
          data.append(convert(char_line, seg_line))
        if max_id is not None and sent_count > max_id:
          break
    return data

  def count_sents(self, filename):
    return super(SegmentationTextReader, self).count_sents(filename[0])

class ContVecReader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".npz" file, which consists of multiply ".npy" files, each
  corresponding to a single sequence of continuous features. This can be
  created in two ways:
  * Use the builtin function numpy.savez_compressed()
  * Create a bunch of .npy files, and run "zip" on them to zip them into an archive.

  The file names should be named XXX_0, XXX_1, etc., where the final number after the underbar
  indicates the order of the sequence in the corpus. This is done automatically by
  numpy.savez_compressed(), in which case the names will be arr_0, arr_1, etc.

  Each numpy file will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,word_ind] if transpose=False
  * sents[sent_id][word_ind,feat_ind] if transpose=True
  """
  yaml_tag = u"!ContVecReader"

  def __init__(self, transpose=False):
    self.transpose = transpose

  def read_sents(self, filename, filter_ids=None):
    npzFile = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      npzKeys = [npzKeys[i] for i in filter_ids]
    for idx, key in enumerate(npzKeys):
      inp = npzFile[key]
      if self.transpose:
        inp = inp.transpose()
      if idx % 1000 == 999:
        logger.info("Read {} lines ({:.2f}%) of {} at {}".format(idx+1, float(idx+1)/len(npzKeys)*100, filename, key))
      yield ArrayInput(inp)
    npzFile.close()

  def count_sents(self, filename):
    npzFile = np.load(filename, mmap_mode="r") # for counting sentences, only read the index
    l = len(npzFile.files)
    npzFile.close()
    return l

class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems)
  """
  yaml_tag = u"!IDReader"

  def read_sents(self, filename, filter_ids=None):
    return map(lambda l: int(l.strip()), self.iterate_filtered(filename, filter_ids))

class TreeInput(Input):
  """
  A tree input, represented as a list of rule indices and other information
  each item in words is a list of data
  """
  def __init__(self, words):
    self.words = words
  def __len__(self):
    return len(self.words)
  def __getitem__(self, key):
    return self.words[key]
  def get_padded_sent(self, token, pad_len):
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([[token] * 6] * pad_len)
    return TreeInput(new_words)
  def __str__(self):
    return " ".join(six.moves.map(str, self.words))

class TreeReader(BaseTextReader, Serializable):
  """
  Reads in parse tree file, with one line per
  parse tree. The vocab object has to be a RuleVocab
  """
  yaml_tag = u'!TreeReader'
  def __init__(self, vocab=None, word_vocab=None, binarize=True, del_preterm_POS=False,
               read_word=False, merge=False, merge_level=-1, add_eos=True, replace_pos=False, bpe_post=False):
    self.vocab = vocab
    self.binarize = binarize
    self.del_preterm_POS = del_preterm_POS
    self.word_vocab = word_vocab
    self.read_word = read_word
    self.merge = merge
    self.merge_level = merge_level
    self.add_eos = add_eos
    self.replace_pos = replace_pos
    self.bpe_post = bpe_post
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)
    if word_vocab is not None:
      self.word_vocab.freeze()
      self.word_vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = RuleVocab()
    if self.read_word and self.word_vocab is None:
      self.word_vocab = Vocab()
    filename = filename.split(',')
    if len(filename) > 1:
      for line, sent_piece in self.iterate_filtered_double(filename[0], filename[1], filter_ids):
        tree = Tree(parse_root(tokenize(line)))
        if self.replace_pos:
          replace_POS(tree.root)
        else:
          if self.del_preterm_POS:
            remove_preterminal_POS(tree.root)
        if self.bpe_post:
          pieces = sent_piece_segs_bpe(sent_piece)
        else:
          pieces = sent_piece_segs(sent_piece)
        split_sent_piece(tree.root, pieces, 0)
        if self.merge:
          merge_tags(tree.root)
        if self.merge_level > 0:
          merge_depth(tree.root, self.merge_level, 0)
        # add x after bpe
        if self.word_vocab:
          add_preterminal_wordswitch(tree.root, self.add_eos)
          if self.binarize:
            tree.root = right_binarize(tree.root, self.read_word)
        else:
          if self.binarize:
            tree.root = right_binarize(tree.root, self.read_word)
          add_preterminal(tree.root)
        tree.reset_timestep()
        yield TreeInput(tree.get_data_root(self.vocab, self.word_vocab))
    else:
      for line in self.iterate_filtered(filename[0], filter_ids):
        tree = Tree(parse_root(tokenize(line)))
        if self.del_preterm_POS:
          remove_preterminal_POS(tree.root)
        if self.merge:
          merge_tags(tree.root)
        if self.merge_level:
          merge_depth(tree.root, self.merge_level, 0)
        if self.word_vocab:
          add_preterminal_wordswitch(tree.root, self.add_eos)
          if self.binarize:
            tree.root = right_binarize(tree.root, self.read_word)
        else:
          if self.binarize:
            tree.root = right_binarize(tree.root, self.read_word)
          add_preterminal(tree.root)
        tree.reset_timestep()
        yield TreeInput(tree.get_data_root(self.vocab, self.word_vocab))
  def freeze(self):
    if self.word_vocab:
      self.word_vocab.freeze()
      self.word_vocab.set_unk(Vocab.UNK_STR)
      self.overwrite_serialize_param('word_vocab', self.word_vocab)
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param("vocab", self.vocab)
    self.vocab.tag_vocab.freeze()
    self.vocab.tag_vocab.set_unk(Vocab.UNK_STR)
    self.vocab.overwrite_serialize_param("tag_vocab", self.vocab.tag_vocab)
  def vocab_size(self):
    return len(self.vocab)
  def tag_vocab_size(self):
    return len(self.vocab.tag_vocab)
  def word_vocab_size(self):
    return len(self.word_vocab)
  def iterate_filtered_double(self, file1, file2, filter_ids=None):
    """
  :param filename: data file (text file)
  :param filter_ids:
  :returns: iterator over lines as strings (useful for subclasses to implement read_sents)
  """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with io.open(file1, encoding='utf-8') as f1:
      with io.open(file2, encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):
          if filter_ids is None or sent_count in filter_ids:
            yield line1, line2
          sent_count += 1
          if max_id is not None and sent_count > max_id:
            break

class TreeNode(object):
  """A class that represents a tree node object """
  def __init__(self, string, children, timestep=-1, id=-1, last_word_t=0):
    self.label = string
    self.children = children
    for c in children:
      if hasattr(c, "set_parent"):
        c.set_parent(self)
    self._parent = None
    self.timestep = timestep
    self.id = id
    self.last_word_t = last_word_t
    self.frontir_label = None
  def is_preterminal(self):
    # return len(self.children) == 1 and (not hasattr(self.children[0], 'is_preterminal'))
    for c in self.children:
      if hasattr(c, 'is_preterminal'): return False
    return True
  def to_parse_string(self):
    c_str = []
    stack = [self]
    while stack:
      cur = stack.pop()
      while not hasattr(cur, 'label'):
        c_str.append(cur)
        if not stack: break
        cur = stack.pop()
      if not hasattr(cur, 'children'): break
      stack.append(u')')
      for c in reversed(cur.children):
        stack.append(c)
      stack.append(u'({} '.format(cur.label))
    return u"".join(c_str)
  def to_string(self, piece=True):
    """
    convert subtree into the sentence it represents
    """
    toks = []
    stack = [self]
    while stack:
      cur = stack.pop()
      while not hasattr(cur, 'label'):
        toks.append(cur)
        if not stack: break
        cur = stack.pop()
      if not hasattr(cur, 'children'): break
      for c in reversed(cur.children):
        stack.append(c)
    if not piece:
      return u" ".join(toks)
    else:
      return u"".join(toks).replace(u'\u2581', u' ').strip()
  def parent(self):
    return self._parent
  def set_parent(self, new_parent):
    self._parent = new_parent
  def add_child(self, child, id2n=None, last_word_t=0):
    self.children.append(child)
    if hasattr(child, "set_parent"):
      child._parent = self
      child.last_word_t = last_word_t
      if id2n:
        child.id = len(id2n)
        id2n[child.id] = child
        return child.id
    return -1
  def copy(self):
    new_node = TreeNode(self.label, [])
    for c in self.children:
      if hasattr(c, 'copy'):
        new_node.add_child(c.copy())
      else:
        new_node.add_child(c)
    return new_node
  def frontir_nodes(self):
    frontir = []
    for c in self.children:
      if hasattr(c, 'children'):
        if len(c.children) == 0:
          frontir.append(c)
        else:
          frontir.extend(c.frontir_nodes())
    return frontir
  def leaf_nodes(self):
    leaves = []
    for c in self.children:
      if hasattr(c, 'children'):
        leaves.extend(c.leaf_nodes())
      else:
        leaves.append(c)
    return leaves
  def get_leaf_lens(self, len_dict):
    if self.is_preterminal():
      l = self.leaf_nodes()
      # if len(l) > 10:
      #    print l, len(l)
      len_dict[len(l)] += 1
      return
    for c in self.children:
      if hasattr(c, 'is_preterminal'):
        c.get_leaf_lens(len_dict)
  def set_timestep(self, t, t2n=None, id2n=None, last_word_t=0, sib_t=0, open_stack=[]):
    """
    initialize timestep for each node
    """
    self.timestep = t
    self.last_word_t = last_word_t
    self.sib_t = sib_t
    next_word_t = last_word_t
    if not t2n is None:
      assert self.timestep == len(t2n)
      assert t not in t2n
      t2n[t] = self
    if not id2n is None:
      self.id = t
      id2n[t] = self
    sib_t = 0
    assert self.label == open_stack[-1]
    open_stack.pop()
    new_open_label = []
    for c in self.children:
      if hasattr(c, 'set_timestep'):
        new_open_label.append(c.label)
    new_open_label.reverse()
    open_stack.extend(new_open_label)
    if open_stack:
      self.frontir_label = open_stack[-1]
    else:
      self.frontir_label = Vocab.ES_STR
    c_t = t
    for c in self.children:
      # c_t = t + 1  # time of current child
      if hasattr(c, 'set_timestep'):
        c_t = t + 1
        t, next_word_t = c.set_timestep(c_t, t2n, id2n, next_word_t, sib_t, open_stack)
      else:
        next_word_t = t
      sib_t = c_t
    return t, next_word_t

class Tree(object):
  """A class that represents a parse tree"""
  yaml_tag = u"!Tree"
  def __init__(self, root=None, sent_piece=None, binarize=False):
    self.id2n = {}
    self.t2n = {}
    self.open_nonterm_ids = []
    self.last_word_t = -1
    if root:
      self.root = TreeNode('XXX', [root])
    else:
      self.last_word_t = 0
      self.root = TreeNode('XXX', [], id=0, timestep=0)
      self.id2n[0] = self.root
  def reset_timestep(self):
    self.root.set_timestep(0, self.t2n, self.id2n, open_stack=['XXX'])
  def __str__(self):
    return self.root.to_parse_string()
  def to_parse_string(self):
    return self.root.to_parse_string()
  def copy(self):
    '''Return a deep copy of the current tree'''
    copied_tree = Tree()
    copied_tree.id2n = {}
    copied_tree.t2n = {}
    copied_tree.open_nonterm_ids = self.open_nonterm_ids[:]
    copied_tree.last_word_t = self.last_word_t
    root = TreeNode('trash', [])
    stack = [self.root]
    copy_stack = [root]
    while stack:
      cur = stack.pop()
      copy_cur = copy_stack.pop()
      copy_cur.label = cur.label
      copy_cur.children = []
      copy_cur.id = cur.id
      copy_cur.timestep = cur.timestep
      copy_cur.last_word_t = cur.last_word_t
      copied_tree.id2n[copy_cur.id] = copy_cur
      if copy_cur.timestep >= 0:
        copied_tree.t2n[copy_cur.timestep] = copy_cur
      for c in cur.children:
        if hasattr(c, 'set_parent'):
          copy_c = TreeNode(c.label, [])
          copy_cur.add_child(copy_c)
          stack.append(c)
          copy_stack.append(copy_c)
        else:
          copy_cur.add_child(c)
    copied_tree.root = root
    return copied_tree
  @classmethod
  def from_rule_deriv(cls, derivs, wordswitch=True):
    tree = Tree()
    stack_tree = [tree.root]
    for x in derivs:
      r, stop = x
      p_tree = stack_tree.pop()
      if type(r) != Rule:
        if p_tree.label != '*':
          for i in derivs:
            if type(i[0]) != Rule:
              print
              i[0].encode('utf-8'), i[1]
            else:
              print
              i[0], i[1]
        assert p_tree.label == '*', p_tree.label
        if wordswitch:
          if r != Vocab.ES_STR:
            p_tree.add_child(r)
            stack_tree.append(p_tree)
        else:
          p_tree.add_child(r)
          if not stop:
            stack_tree.append(p_tree)
        continue
      if p_tree.label == 'XXX':
        new_tree = TreeNode(r.lhs, [])
        p_tree.add_child(new_tree)
      else:
        if p_tree.label != r.lhs:
          for i in derivs:
            if type(i[0]) != Rule:
              print
              i[0].encode('utf-8'), i[1]
            else:
              print
              i[0], i[1]
          print
          tree.to_parse_string().encode('utf-8')
          print
          p_tree.label.encode('utf-8'), r.lhs.encode('utf-8')
          exit(1)
        assert p_tree.label == r.lhs, "%s %s" % (p_tree.label, r.lhs)
        new_tree = p_tree
      open_nonterms = []
      for child in r.rhs:
        if child not in r.open_nonterms:
          new_tree.add_child(child)
        else:
          n = TreeNode(child, [])
          new_tree.add_child(n)
          open_nonterms.append(n)
      open_nonterms.reverse()
      stack_tree.extend(open_nonterms)
    return tree
  def to_string(self, piece=False):
    """
    convert subtree into the sentence it represents
    """
    return self.root.to_string(piece)
  def add_rule(self, id, rule):
    ''' Add one node to the tree based on current rule; only called on root tree '''
    node = self.id2n[id]
    node.set_timestep(len(self.t2n), self.t2n)
    node.last_word_t = self.last_word_t
    assert rule.lhs == node.label, "Rule lhs %s does not match the node %s to be expanded" % (rule.lhs, node.label)
    new_open_ids = []
    for rhs in rule.rhs:
      if rhs in rule.open_nonterms:
        new_open_ids.append(self.id2n[id].add_child(TreeNode(rhs, []), self.id2n))
      else:
        self.id2n[id].add_child(rhs)
        self.last_word_t = node.timestep
    new_open_ids.reverse()
    self.open_nonterm_ids.extend(new_open_ids)
    if self.open_nonterm_ids:
      node.frontir_label = self.id2n[self.open_nonterm_ids[-1]].label
    else:
      node.frontir_label = None
  def get_next_open_node(self):
    if len(self.open_nonterm_ids) == 0:
      print("stack empty, tree is complete")
      return -1
    return self.open_nonterm_ids.pop()
  def get_timestep_data(self, id):
    ''' Return a list of timesteps data associated with current tree node; only called on root tree '''
    data = []
    if self.id2n[id].parent():
      data.append(self.id2n[id].parent().timestep)
    else:
      data.append(0)
    data.append(self.id2n[id].last_word_t)
    return data
  def get_data_root(self, rule_vocab, word_vocab=None):
    data = []
    for t in range(1, len(self.t2n)):
      node = self.t2n[t]
      children, open_nonterms = [], []
      for c in node.children:
        if type(c) == str:
          children.append(c)
        else:
          children.append(c.label)
          open_nonterms.append(c.label)
      paren_t = 0 if not node.parent() else node.parent().timestep
      is_terminal = 1 if len(open_nonterms) == 0 else 0
      if word_vocab and is_terminal:
        leaf_len = len(node.children)
        is_first = True
        for c in node.children:
          d = [word_vocab.convert(c), paren_t, node.last_word_t, is_terminal, 0, leaf_len, is_first]
          data.append(d)
          leaf_len = -1
          is_first = False
        data[-1][4] = 1
      else:
        r = Rule(node.label, children, open_nonterms)
        d = [rule_vocab.convert(Rule(node.label, children, open_nonterms)), paren_t,
             node.last_word_t, is_terminal, None, None, False]
             #rule_vocab.tag_vocab.convert(node.frontir_label), rule_vocab.tag_vocab.convert(node.label), False]
        data.append(d)
    return data
  def get_bpe_rule(self, rule_vocab):
    ''' Get the rules for doing bpe. Label left and right child '''
    rule_idx = []
    for t in range(1, len(self.t2n)):
      node = self.t2n[t]
      children, open_nonterms = [], []
      child_idx = 1
      attach_tag = len(children) > 1
      for c in node.children:
        if type(c) == str:
          if attach_tag:
            children.append('{}_{}'.format(c, child_idx))
          else:
            children.append(c)
        else:
          if attach_tag:
            children.append('{}_{}'.format(c.label, child_idx))
          else:
            children.append(c.label)
          open_nonterms.append(c.label)
        child_idx += 1
      r = rule_vocab.convert(Rule(node.label, children, open_nonterms))
      rule_idx.append(r)
    return rule_idx
  def query_open_node_label(self):
    return self.id2n[self.open_nonterm_ids[-1]].label
def sent_piece_segs(p):
  '''
  Segment a sentence piece string into list of piece string for each word
  '''
  toks = re.compile(r'\u2581')
  ret = []
  p_start = 0
  for m in toks.finditer(p):
    pos = m.start()
    if pos == 0:
      continue
    ret.append(p[p_start:pos])
    p_start = pos
  if p_start != len(p) - 1:
    ret.append(p[p_start:])
  return ret

def sent_piece_segs_bpe(p):
  '''
Segment a sentence piece string into list of piece string for each word
'''
  # print p
  # print p.split()
  # toks = re.compile(ur'\xe2\x96\x81[^(\xe2\x96\x81)]+')
  toks = p.split()
  ret = []
  cur = []
  for t in toks:
    cur.append(t)
    if not t.endswith(u'@@'):
      ret.append(u' '.join(cur))
      cur = []
  return ret

def sent_piece_segs_post(p):
  '''
Segment a sentence piece string into list of piece string for each word
'''
  # print p
  # print p.split()
  # toks = re.compile(ur'\xe2\x96\x81[^(\xe2\x96\x81)]+')
  toks = re.compile(r'\u2581')
  ret = []
  p_start = 0
  for m in toks.finditer(p):
    pos = m.start()
    if pos == 0:
      continue
    ret.append(p[p_start:pos + 1].strip())
    p_start = pos + 1
  if p_start != len(p) - 1:
    ret.append(p[p_start:])
  return ret

def split_sent_piece(root, piece_l, word_idx):
  '''
  Split words into sentence piece
  '''
  new_children = []
  for i, c in enumerate(root.children):
    if type(c) == str:
      piece = piece_l[word_idx].split()
      word_idx += 1
      new_children.extend(piece)
    else:
      word_idx = split_sent_piece(c, piece_l, word_idx)
      new_children.append(c)
  root.children = new_children
  return word_idx

def right_binarize(root, read_word=False):
  '''
  Right binarize a CusTree object
  read_word: if true, do not binarize terminal rules
  '''
  if type(root) == str:
    return root
  if read_word and root.label == u'*':
    return root
  if len(root.children) <= 2:
    new_children = []
    for c in root.children:
      new_children.append(right_binarize(c))
    root.children = new_children
  else:
    if "__" in root.label:
      new_label = root.label
    else:
      new_label = root.label + "__"
    n_left_child = TreeNode(new_label, root.children[1:])
    n_left_child._parent = root
    root.children = [right_binarize(root.children[0]), right_binarize(n_left_child)]
  return root

def add_preterminal(root):
  ''' Add preterminal X before each terminal symbol '''
  for i, c in enumerate(root.children):
    if type(c) == str:
      n = TreeNode(u'*', [c])
      n.set_parent(root)
      root.children[i] = n
    else:
      add_preterminal(c)

def add_preterminal_wordswitch(root, add_eos):
  ''' Add preterminal X before each terminal symbol '''
  ''' word_switch: one * symbol for each phrase chunk
      preterm_paren: * preterm parent already created
  '''
  preterm_paren = None
  new_children = []
  if root.label == u'*':
    if add_eos:
      root.add_child(Vocab.ES_STR)
    return root
  for i, c in enumerate(root.children):
    if type(c) == str:
      if not preterm_paren:
        preterm_paren = TreeNode('*', [])
        preterm_paren.set_parent(root)
        new_children.append(preterm_paren)
      preterm_paren.add_child(c)
    else:
      if preterm_paren and add_eos:
        preterm_paren.add_child(Vocab.ES_STR)
      c = add_preterminal_wordswitch(c, add_eos)
      new_children.append(c)
      preterm_paren = None
  if preterm_paren and add_eos:
    preterm_paren.add_child(Vocab.ES_STR)
  root.children = new_children
  return root

def remove_preterminal_POS(root):
  ''' Remove the POS tag before terminal '''
  for i, c in enumerate(root.children):
    if c.is_preterminal():
      root.children[i] = c.children[0]
    else:
      remove_preterminal_POS(c)
def replace_POS(root):
  ''' simply replace POS with * '''
  for i, c in enumerate(root.children):
    if c.is_preterminal():
      c.label = '*'
    else:
      replace_POS(c)
def merge_depth(root, max_depth, cur_depth):
  ''' raise up trees whose depth exceed max_depth '''
  if cur_depth >= max_depth:
    # root.label = u'*'
    root.children = root.leaf_nodes()
    return root
  new_children = []
  for i, c in enumerate(root.children):
    if hasattr(c, 'children'):
      c = merge_depth(c, max_depth, cur_depth + 1)
      # combine consecutive * nodes
      if new_children and hasattr(new_children[-1], 'label') and new_children[
        -1].is_preterminal() and c.is_preterminal():
        for x in c.children:
          new_children[-1].add_child(x)
      else:
        new_children.append(c)
    else:
      new_children.append(c)
  root.children = new_children
  return root
def merge_tags(root):
  ''' raise up trees whose label is in a given set '''
  kept_label = set([u'np', u'vp', u'pp', u's', u'root', u'sbar', u'sinv', u'XXX', u'prn', u'adjp', u'advp',
                    u'whnp', u'whadvp',
                    u'NP', u'VP', u'PP', u'S', u'ROOT', u'SBAR', u'FRAG', u'SINV', u'PRN'])
  if not root.label in kept_label:
    root.label = u'xx'
  for i, c in enumerate(root.children):
    if hasattr(c, 'children'):
      c = merge_tags(c)
    root.children[i] = c
  return root
def combine_tags(root):
  tag_dict = {'adjp': 'advp', 'sq': 'sbarq', 'whadjp': 'whadvp'}
# Tokenize a string.
# Tokens yielded are of the form (type, string)
# Possible values for 'type' are '(', ')' and 'WORD'
def tokenize(s):
  toks = re.compile(r' +|[^() ]+|[()]')
  for match in toks.finditer(s):
    s = match.group(0)
    if s[0] == ' ':
      continue
    if s[0] in '()':
      yield (s, s)
    else:
      yield ('WORD', s)
# Parse once we're inside an opening bracket.
def parse_inner(toks):
  ty, name = next(toks)
  if ty != 'WORD': raise ParseError
  children = []
  while True:
    ty, s = next(toks)
    # print ty, s
    if ty == '(':
      children.append(parse_inner(toks))
    elif ty == ')':
      return TreeNode(name, children)
    else:
      children.append(s)
class ParseError(Exception):
  pass
# Parse this grammar:
# ROOT ::= '(' INNER
# INNER ::= WORD ROOT* ')'
# WORD ::= [A-Za-z]+
def parse_root(toks):
  ty, s = next(toks)
  if ty != '(':
    # print ty, s
    raise ParseError
  return parse_inner(toks)


###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader, trg_reader, src_file, trg_file, ref_len_file=None,
                         batcher=None, sample_sents=None, max_num_sents=None, max_src_len=None, max_trg_len=None):
  '''
  A utility function to read a parallel corpus.

  :returns: A tuple of (src_data, trg_data, src_batches, trg_batches) where *_batches = *_data if batcher=None
  '''
  src_data = []
  trg_data = []
  if ref_len_file:
    trg_len_data = []
    ref_len_nums = []
    with open(ref_len_file, encoding='utf-8') as fp:
      for line in fp:
        ref_len_nums.append(line)
  if sample_sents:
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
    filter_ids = None
    src_len, trg_len = 0, 0
  src_train_iterator = src_reader.read_sents(src_file, filter_ids)
  trg_train_iterator = trg_reader.read_sents(trg_file, filter_ids)
  i = 0
  for src_sent, trg_sent in six.moves.zip_longest(src_train_iterator, trg_train_iterator):
    if src_sent is None or trg_sent is None:
      raise RuntimeError(f"training src sentences don't match trg sentences: {src_len or src_reader.count_sents(src_file)} != {trg_len or trg_reader.count_sents(trg_file)}!")
    if max_num_sents and (max_num_sents <= len(src_data)):
      break
    src_len_ok = max_src_len is None or len(src_sent) <= max_src_len
    trg_len_ok = max_trg_len is None or len(trg_sent) <= max_trg_len
    if src_len_ok and trg_len_ok:
      src_data.append(src_sent)
      trg_data.append(trg_sent)
      if ref_len_file:
        n = len(ref_len_nums[i].split())
        trg_len_data.append(n)
    i+=1
  # Pack batches
  if batcher != None:
    if ref_len_file:
     src_batches, trg_batches, trg_len_batches = batcher.pack(src_data, trg_data, trg_len=trg_len_data)
    else:
      src_batches, trg_batches = batcher.pack(src_data, trg_data)
      trg_len_batches = None
  else:
    src_batches, trg_batches = src_data, trg_data
    if ref_len_file:
      trg_len_batches = trg_len_data
    else:
      trg_len_batches = None
  return src_data, trg_data, src_batches, trg_batches, trg_len_batches