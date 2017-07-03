import numpy as np
import itertools
import os
from collections import defaultdict
from six.moves import zip
from serializer import Serializable
from vocab import *
import model_globals

###### Classes representing single inputs

class Input:
  """
  A template class to represent all inputs.
  """
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Input subclasses")
  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")
  def get_padded_sent(self, token, pad_len):
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sent, represented as a list of tokens
  """
  def __init__(self, l):
    self.l = l
  def __len__(self):
    return self.l.__len__()
  def __getitem__(self, key):
    return self.l.__getitem__(key)
  def get_padded_sent(self, token, pad_len):
    self.l.extend([token] * pad_len)
    return self
    
class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens 
  """
  def __init__(self, nparr):
    self.nparr = nparr
  def __len__(self):
    return self.nparr.__len__()
  def __getitem__(self, key):
    return self.nparr.__getitem__(key)
  def get_padded_sent(self, token, pad_len):
    if pad_len>0:
      self.nparr = np.append(self.nparr, np.repeat(token.reshape(1,len(token)), pad_len, axis=0), axis=0)
    return self
  def get_array(self):
    return self.nparr

###### Classes that will read in a file and turn it into an input

class InputReader:

  def read_file(self, filename, max_num=None):
    raise RuntimeError("Input readers must implement the read_file function")

  def freeze(self):
    pass


class PlainTextReader(InputReader, Serializable):
  """
  Handles the typical case of reading plain text files,
  with one sent per line.
  """
  yaml_tag = u'!PlainTextReader'
  def __init__(self, vocab=None):
    self.vocab = vocab

  def read_file(self, filename, max_num=None):
    if self.vocab is None:
      self.vocab = Vocab()
    sents = []
    with open(filename) as f:
      for line in f:
        words = line.decode('utf-8').strip().split()
        sent = [self.vocab.convert(word) for word in words]
        sent.append(self.vocab.convert(Vocab.ES_STR))
        sents.append(SimpleSentenceInput(sent))
        if max_num is not None and len(sents) >= max_num:
          break
    return sents

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.serialize_params["vocab"] = self.vocab

    
class ContVecReader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.
  
  We assume a list of matrices (sents) serialized as .npz (with numpy.savez_compressed())
  Sentences should be named arr_0, arr_1, ... (=np default for unnamed archives).
  We can index them as sents[sent_no][word_ind,feat_ind]
  """
  yaml_tag = u"!ContVecReader"
  
  def __init__(self):
    pass

  def read_file(self, filename, max_num=None):
    npzFile = np.load(filename)
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[1]))
    if max_num is not None and max_num < len(npzKeys):
      npzKeys = npzKeys[:max_num]
    sents = map(lambda f:ArrayInput(npzFile[f]), npzKeys)
    npzFile.close()
    return sents

###### CorpusParser

class CorpusParser:
  """A class that can read in corpora for training and testing"""

  def read_training_corpus(self, training_corpus):
    """Read in the training corpus"""
    raise RuntimeError("CorpusParsers must implement read_training_corpus to read in the training/dev corpora")


class BilingualCorpusParser(CorpusParser, Serializable):
  """A class that reads in bilingual corpora, consists of two InputReaders"""

  yaml_tag = u"!BilingualCorpusParser"
  def __init__(self, src_reader, trg_reader):
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.serialize_params = {"src_reader": src_reader, "trg_reader": trg_reader}

  def read_training_corpus(self, training_corpus):
    training_corpus.train_src_data = self.src_reader.read_file(training_corpus.train_src)
    training_corpus.train_trg_data = self.trg_reader.read_file(training_corpus.train_trg)
    self.src_reader.freeze()
    self.trg_reader.freeze()
    training_corpus.dev_src_data = self.src_reader.read_file(training_corpus.dev_src)
    training_corpus.dev_trg_data = self.trg_reader.read_file(training_corpus.dev_trg)


###### Obsolete Functions

# TODO: The following doesn't follow the current API. If it is necessary, it should be retooled
# class MultilingualAlignedCorpusReader(object):
#     """Handles the case of reading TED talk files
#     """
#     
#     def __init__(self, corpus_path, vocab=None, delimiter='\t', trg_token=True, bilingual=True,
#                  lang_dict={'src': ['fr'], 'trg': ['en']}, zero_shot=False, eval_lang_dict=None):
#         
#         self.empty_line_flag = '__NULL__'
#         self.corpus_path = corpus_path
#         self.delimiter = delimiter
#         self.bilingual = bilingual    
#         self.lang_dict = lang_dict
#         self.lang_set = set()
#         self.trg_token = trg_token
#         self.zero_shot = zero_shot
#         self.eval_lang_dict = eval_lang_dict
# 
#         for list_ in self.lang_dict.values():
#             for lang in list_:
#                 self.lang_set.add(lang)
#         
#         self.data = dict()
#         self.data['train'] = self.read_aligned_corpus(split_type='train')
#         self.data['test'] = self.read_aligned_corpus(split_type='test')
#         self.data['dev'] = self.read_aligned_corpus(split_type='dev')
#     
#     
#     def read_data(self, file_loc_):
#         data_list = list()
#         with open(file_loc_) as fp:
#             for line in fp:
#                 try:
#                     text = line.strip()
#                 except IndexError:
#                     text = self.empty_line_flag
#                 data_list.append(text)
#         return data_list
#     
#     
#     def filter_text(self, dict_):
#         if self.trg_token:
#             field_index = 1
#         else:
#             field_index = 0
#         data_dict = defaultdict(list)
#         list1 = dict_['src']
#         list2 = dict_['trg']
#         for sent1, sent2 in zip(list1, list2):
#             try:
#                 src_sent = ' '.join(sent1.split()[field_index: ])
#             except IndexError:
#                 src_sent = '__NULL__'
#             
#             if src_sent.find(self.empty_line_flag) != -1: 
#                 continue
#             
#             elif sent2.find(self.empty_line_flag) != -1:
#                 continue
#             
#             else:
#                 data_dict['src'].append(sent1)
#                 data_dict['trg'].append(sent2)
#         return data_dict
#     
#     
#     def read_file(self, split_type, data_type):
#         return self.data[split_type][data_type]
#         
#     
#     def save_file(self, path_, split_type, data_type):
#         with open(path_, 'w') as fp:
#             for line in self.data[split_type][data_type]:
#                 fp.write(line + '\n')
#     
#     
#     def add_trg_token(self, list_, lang_id):
#         new_list = list()
#         token = '__' + lang_id + '__'
#         for sent in list_:
#             new_list.append(token + ' ' + sent)
#         return new_list
#     
#     def read_aligned_corpus(self, split_type='train'):
#         
#         split_type_path = os.path.join(self.corpus_path, split_type)
#         data_dict = defaultdict(list)
# 
#         if self.zero_shot:
#             if split_type == "train":
#                 iterable = zip(self.lang_dict['src'], self.lang_dict['trg'])
#             else:
#                 iterable = zip(self.eval_lang_dict['src'], self.eval_lang_dict['trg'])
# 
#         elif self.bilingual:
#             iterable = itertools.product(self.lang_dict['src'], self.lang_dict['trg'])
# 
#         for s_lang, t_lang in iterable:
#                 for talk_dir in os.listdir(split_type_path):
#                     dir_path = os.path.join(split_type_path, talk_dir)
# 
#                     talk_lang_set = set([l.split('.')[0] for l in os.listdir(dir_path)])
# 
#                     if s_lang not in talk_lang_set or t_lang not in talk_lang_set:
#                         continue
# 
#                     for infile in os.listdir(dir_path):
#                         lang = os.path.splitext(infile)[0]
# 
#                         if lang in self.lang_set:
#                             file_path = os.path.join(dir_path, infile)
#                             text = self.read_data(file_path)
#                             
#                             if lang == s_lang:
#                                 if self.trg_token:
#                                     text = self.add_trg_token(text, t_lang)
#                                     data_dict['src'] += text
#                                 else:
#                                     data_dict['src'] += text
#                             
#                             elif lang == t_lang:
#                                 data_dict['trg'] += text
#         
#         new_data_dict = self.filter_text(data_dict)    
#         return new_data_dict
#     
#     
# if __name__ == "__main__":
# 
#     # Testing the code
#     data_path = "/home/devendra/Desktop/Neural_MT/scrapped_ted_talks_dataset/web_data_temp"
#     zs_train_lang_dict={'src': ['pt-br', 'en'], 'trg': ['en', 'es']}
#     zs_eval_lang_dict = {'src': ['pt-br'], 'trg': ['es']}
#     
#     obj = MultilingualAlignedCorpusReader(corpus_path=data_path, lang_dict=zs_train_lang_dict, trg_token=True,
#                                           eval_lang_dict=zs_eval_lang_dict, zero_shot=True, bilingual=False)
# 
# 
#     #src_test_list = obj.read_file(split_type='test', data_type='src')
#     #trg_test_list = obj.read_file(split_type='test', data_type='trg')
#     
#     #print len(src_test_list)
#     #print len(trg_test_list)
#     
#     #for sent_s, sent_t in zip(src_test_list, trg_test_list):
#     #    print sent_s, "\t", sent_t
#         
#     obj.save_file("../ted_sample/zs_s.train", split_type='train', data_type='src')
#     obj.save_file("../ted_sample/zs_t.train", split_type='train', data_type='trg')
#     
#     obj.save_file("../ted_sample/zs_s.test", split_type='test', data_type='src')
#     obj.save_file("../ted_sample/zs_t.test", split_type='test', data_type='trg')
#     
#     obj.save_file("../ted_sample/zs_s.dev", split_type='dev', data_type='src')
#     obj.save_file("../ted_sample/zs_t.dev", split_type='dev', data_type='trg')
