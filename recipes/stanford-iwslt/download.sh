#!/bin/bash

wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi

tail -n +4 vocab.en > vocab.en.xnmt
tail -n +4 vocab.vi > vocab.vi.xnmt

