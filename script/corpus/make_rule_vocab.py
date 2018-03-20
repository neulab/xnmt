import sys
from xnmt.input import *
from xnmt.vocab import *

vocab = set()
filename = "examples/data/dev-head5.parse.en,examples/data/dev-head5.piece.en"
rule_vocab_file = "examples/data/dev-head5.rule.vocab.en"
word_vocab_file = "examples/data/dev-head5.word.vocab.en"
binarize = False
del_preterm_POS=True
replace_pos=False
read_word=True
merge=False
merge_level=-1
add_eos=True
bpe_post=True
tr_reader = TreeReader(binarize=binarize,
                       del_preterm_POS=del_preterm_POS,
                       read_word=read_word,
                       merge=merge,
                       merge_level=merge_level,
                       add_eos=add_eos,
                       replace_pos=replace_pos,
                       bpe_post=bpe_post)

for i in tr_reader.read_sents(filename=filename):
  pass

filter_toks = set([Vocab.ES_STR, Vocab.SS_STR, Vocab.UNK_STR])
with open(rule_vocab_file, 'w', encoding='utf-8') as myfile:
  for r in tr_reader.vocab:
    if r not in filter_toks:
      myfile.write(str(r) + '\n')

if word_vocab_file:
  with open(word_vocab_file, 'w', encoding='utf-8') as myfile:
    for w in tr_reader.word_vocab:
      if w not in filter_toks:
        myfile.write(w + '\n')
