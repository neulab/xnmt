import numpy as np
import os
from collections import defaultdict
from six.moves import zip
from vocab import *


class Input:
  '''
  A template class to represent all inputs.
  '''
  pass

class Sentence:
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Sentence subclasses")
  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Sentence subclasses")
  def get_padded_sentence(self, token, pad_len):
    raise NotImplementedError("get_padded_sentence() must be implemented by Sentence subclasses")

class SimpleSentence(Sentence):
  def __init__(self, l):
    self.l = l
  def __len__(self):
    return self.l.__len__()
  def __getitem__(self, key):
    return self.l.__getitem__(key)
  def get_padded_sentence(self, token, pad_len):
    self.l.extend([token] * pad_len)
    return self
    
class ArraySentence(Sentence):
  def __init__(self, nparr):
    self.nparr = nparr
  def __len__(self):
    return self.nparr.__len__()
  def __getitem__(self, key):
    return self.nparr.__getitem__(key)
  def get_padded_sentence(self, token, pad_len):
    if pad_len>0:
      self.nparr = np.append(self.nparr, np.repeat(token.reshape(1,len(token)), pad_len, axis=0), axis=0)
    return self
  def get_array(self):
    return self.nparr

class InputReader:
  @staticmethod
  def create_input_reader(file_format, vocab=None):
    if file_format == "text":
      return PlainTextReader(vocab)
    elif file_format == "contvec":
      return ContVecReader()
    else:
      raise RuntimeError("Unkonwn input type {}".format(file_format))


class PlainTextReader(InputReader):
  '''
  Handles the typical case of reading plain text files,
  with one sentence per line.
  '''
  def __init__(self, vocab=None):
    if vocab is None:
      self.vocab = Vocab()
    else:
      self.vocab = vocab

  def read_file(self, filename):
    sentences = []
    with open(filename) as f:
      for line in f:
        words = line.strip().split()
        sentence = [self.vocab.convert(word) for word in words]
        sentence.append(self.vocab.convert('</s>'))
        sentences.append(SimpleSentence(sentence))
    return sentences

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk('<unk>')

    
class ContVecReader(InputReader):
  '''
  Handles the case where sentences are sequences of continuous-space vectors.
  We assume a list of matrices (sentences) serialized as .npz (with numpy.savez_compressed())
  We can index them as sentences[sent_no][word_ind,feat_ind]
  '''
  def __init__(self):
    self.vocab = Vocab()

  def read_file(self, filename):
    npzFile = np.load(filename)
    sentences = map(lambda f:ArraySentence(npzFile[f]), npzFile.files)
    npzFile.close()
    return sentences

  def freeze(self):
    pass



class MultilingualAlignedCorpusReader(object):
    """Handles the case of reading TED talk files
    """
    
    def __init__(self, corpus_path, vocab=None, delimiter='\t', target_token=True,
                 bilingual=True, lang_dict={'source': ['fr'], 'target': ['en']}):
        
        self.empty_line_flag = ''
        self.corpus_path = corpus_path
        self.delimiter = delimiter
        self.bilingual = bilingual    
        self.lang_dict = lang_dict
        self.lang_set = set()
        self.target_token = target_token
        
        for list_ in self.lang_dict.values():
            for lang in list_:
                self.lang_set.add(lang)
        
        self.data = dict()
        self.data['train'] = self.read_aligned_corpus(split_type='train')
        self.data['test'] = self.read_aligned_corpus(split_type='test')
        self.data['dev'] = self.read_aligned_corpus(split_type='dev')
    
    
    def read_data(self, file_loc_):
        data_list = list()
        with open(file_loc_) as fp:
            for line in fp:
                try:
                    text = line.strip()
                except IndexError:
                    text = self.empty_line_flag
                data_list.append(text)
        return data_list
    
    
    def filter_text(self, dict_):
        if self.target_token:
            field_index = 1
        else:
            field_index = 0
        data_dict = defaultdict(list)
        list1 = dict_['source']
        list2 = dict_['target']
        for sent1, sent2 in zip(list1, list2):
            if (self.empty_line_flag == sent1.split()[field_index]) or (sent2 == self.empty_line_flag):
                continue
            data_dict['source'].append(sent1)
            data_dict['target'].append(sent2)
        return data_dict
    
    
    def read_file(self, split_type, data_type):
        return self.data[split_type][data_type]
        
    
    def save_file(self, path_, split_type, data_type):
        with open(path_, 'w') as fp:
            for line in self.data[split_type][data_type]:
                fp.write(line + '\n')
    
    
    def add_target_token(self, list_, lang_id):
        new_list = list()
        token = '__' + lang_id + '__'
        for sent in list_:
            new_list.append(token + ' ' + sent)
        return new_list
    
    def read_aligned_corpus(self, split_type='train'):
        
        split_type_path = os.path.join(self.corpus_path, split_type)
        data_dict = defaultdict(list)
        
        for s_lang in self.lang_dict['source']:
            for t_lang in self.lang_dict['target']:
        
                for talk_dir in os.listdir(split_type_path):
                    dir_path = os.path.join(split_type_path, talk_dir)

                    talk_lang_set = set([l.split('.')[0] for l in os.listdir(dir_path)])

                    if s_lang not in talk_lang_set or t_lang not in talk_lang_set:
                        continue

                    for infile in os.listdir(dir_path):
                        lang = os.path.splitext(infile)[0]

                        if lang in self.lang_set:
                            file_path = os.path.join(dir_path, infile)
                            text = self.read_data(file_path)
                            
                            if lang == s_lang:
                                if self.target_token:
                                    text = self.add_target_token(text, t_lang)
                                    data_dict['source'] += text
                                else:
                                    data_dict['source'] += text
                            
                            elif lang == t_lang:
                                data_dict['target'] += text
        
        new_data_dict = self.filter_text(data_dict)    
        return new_data_dict
    
    
if __name__ == "__main__":
    
    # Testing the code
    data_path = "/home/devendra/Desktop/Neural_MT/scrapped_ted_talks_dataset/web_data_temp"
    lang_dict={'source': ['fr'], 'target': ['en']}
    
    obj = MultilingualAlignedCorpusReader(corpus_path=data_path, lang_dict=lang_dict, target_token=True)
    
    #source_test_list = obj.read_file(split_type='test', data_type='source')
    #target_test_list = obj.read_file(split_type='test', data_type='target')
    
    #print len(source_test_list)
    #print len(target_test_list)
    
    #for sent_s, sent_t in zip(source_test_list, target_test_list):
    #    print sent_s, "\t", sent_t
        
    obj.save_file("ted_sample/fr.train", split_type='train', data_type='source')
    obj.save_file("ted_sample/en.train", split_type='train', data_type='target')
    
    obj.save_file("ted_sample/fr.test", split_type='test', data_type='source')
    obj.save_file("ted_sample/en.test", split_type='test', data_type='target')
    
    obj.save_file("ted_sample/fr.dev", split_type='dev', data_type='source')
    obj.save_file("ted_sample/en.dev", split_type='dev', data_type='target')