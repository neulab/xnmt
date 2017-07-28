Preprocessing
=============

In machine translation, and neural MT in particular, properly pre-processing input 
before passing it to the learner can greatly increase translation accuracy.
This document describes the preprocessing options available within ``xnmt``, and
documents where external executables can be plugged into the experiment framework.


Tokenization
------------
A number of tokenization methods are available out of the box; others can be plugged in either
with some help (like sentencepiece) or by passing parameters through the experiment framework
through to the external decoders.

Multiple tokenizers can be run on the same text; for example, it may be (is there a citation?) that 
running the Moses tokenizer before performing Byte-pair encoding (BPE) is preferable to either one or
the other. It is worth noting, however, that if you want to exactly specify your vocabulary size
at tokenization first, an exact-size tokenizer like BPE should be specified (and thus run) *last*. 

1. Sentencepiece:         An extenral tokenizer library that permits a large number of tokenization
                          options, is written in C++, and is very fast. However, it must be installed
                          separately to ``xnmt``. 
                          Specification of the training file is set through the experiment framework,
                          but that (and all other) options can be passed transparently by adding them
                          to the experiment config.
                          See the Sentencepiece section for more specific information on this tokenizer.

2. External Tokenizers:   Any external tokenizer can be used as long as it tokenizes stdin and outputs
                          to stdout. A single Yaml dictionary labelled ``tokenizer_args``
                          is used to pass all (and any) options to the external tokenizer.
                          The option ``detokenizer_path``, and its option dictionary, ``detokenizer_args``,
                          can optionally be used to specify a detokenizer.

.. 3. Byte-Pair Encoding:    A compression-inspired unsupervised sub-word unit encoding
                          that performs well (Sennrich, 2016) and permits specification
                          of an exact vocabulary size. Native to ``xnmt``; written in Python.
                          Invoked with tokenizer type ``bpe``. (TODO)

Sentencepiece
+++++++++++++
The YAML options supported by the SentencepieceTokenizer are almost exactly those presented
in the Sentencepiece readme. Some notable exceptions are below:

 - Instead of ``extra_options``, since one must be able to pass separate options to the 
   encoder and the decoder, use ``encode_extra_options`` and ``decode_extra_options``, respectively.
 - When specifying extra options as above, note that ``eos`` and ``bos`` are both off-limits,
   and will produce odd errors in ``vocab.py``. This is because these options add ``<s>`` and ``</s>``
   to the output, which are already addded by ``xnmt``, and are reserved types.
 - Unfortunately, right now, if tokenizers are chained together we see the following behavior:
     - If the Moses tokenizer is run first, and tokenizes files that are to be used for training BPE
       in Sentencepiece, Sentencepiece will learn off of the *original* files, not the Moses-tokenized
       ones. 


