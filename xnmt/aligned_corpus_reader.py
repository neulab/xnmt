import os
from vocab import *
from collections import defaultdict
from six.moves import zip


class MultilingualAlignedCorpusReader(object):
    """Handles the case of reading TED talk files
    """
    
    def __init__(self, corpus_path, vocab=None, delimiter='\t', 
                 bilingual=True, lang_dict={'source': ['fr'], 'target': ['en']}):
        
        self.corpus_path = corpus_path
        self.delimiter = delimiter
        self.bilingual = bilingual    
        self.lang_dict = lang_dict
        self.lang_set = set()
        
        for list_ in self.lang_dict.values():
            for lang in list_:
                self.lang_set.add(lang)
        
        self.data = dict()
        self.data['train'] = self.read_aligned_corpus(split_type='train')
        self.data['test'] = self.read_aligned_corpus(split_type='test')
        self.data['dev'] = self.read_aligned_corpus(split_type='dev')
    
    
    def read_data(self, file_loc_):
        """Reads the .tsv format file. text is in field 3 ( 0 index based)
        """
        data_list = list()
        with open(file_loc_) as fp:
            for line in fp:
                text = line.strip().split(self.delimiter)[3]
                data_list.append(text)
        return data_list
    
    
    def filter_text(self, dict_):
        """Removes the aligned lines if either one of them has '__NULL__' string
            It signifies empty string 
        """
        data_dict = defaultdict(list)
        list1 = dict_['source']
        list2 = dict_['target']
        for sent1, sent2 in zip(list1, list2):
            if (sent1 == '__NULL__') or (sent2 == '__NULL__'):
                continue
            data_dict['source'].append(sent1)
            data_dict['target'].append(sent2)
        return data_dict
    
    
    def read_file(self, split_type, data_type):
        """ Test function
        """
        return self.data[split_type][data_type]
    
    
    def save_file(self, path_, split_type, data_type):
        """ Saves the data
        """
        with open(path_, 'w') as fp:
            for line in self.data[split_type][data_type]:
                fp.write(line + '\n')
    
    
    def read_aligned_corpus(self, split_type='train'):
        """ Aligned corpus reader
        """
        split_type_path = os.path.join(self.corpus_path, split_type)
        
        read_dict = defaultdict(list)
        data_dict = defaultdict(list)
        
        for talk_dir in os.listdir(split_type_path):
            dir_path = os.path.join(split_type_path, talk_dir)
        
            for infile in os.listdir(dir_path):
                lang = os.path.splitext(infile)[0]
            
                if lang in self.lang_set:
                    file_path = os.path.join(dir_path, infile)
                    read_dict[lang] += self.read_data(file_path)
        
        for k, list_ in self.lang_dict.iteritems():
            for lang in list_:
                data_dict[k] += read_dict[lang]
        
        new_data_dict = self.filter_text(data_dict)    
        return new_data_dict
    
    
if __name__ == "__main__":
    
    # Testing the code
    data_path = "/home/devendra/Desktop/Neural_MT/scrapped_ted_talks_dataset/web_data_aligned"
    obj = MultilingualAlignedCorpusReader(corpus_path=data_path)
    source_test_list = obj.read_file(split_type='test', data_type='source')
    target_test_list = obj.read_file(split_type='test', data_type='target')
    
    print len(source_test_list)
    print len(target_test_list)
    
    for sent_s, sent_t in zip(source_test_list, target_test_list):
        print sent_s, "\t", sent_t
        
    obj.save_file("temp.txt", split_type='test', data_type='source')