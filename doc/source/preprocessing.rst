Preprocessing
=============

In machine translation, and neural MT in particular, properly pre-processing input 
before passing it to the learner can greatly increase translation accuracy.
This document describes the preprocessing options available within ``xnmt``, and
documents where external executables can be plugged into the experiment framework.


Tokenization
------------
A number of tokenization methods are available out of the box; others can be plugged in either
with some help (like [THATGOOGLEONE]) or by passing parameters through the experiment framework
through to the external decoders.

Multiple tokenizers can be run on the same text; for example, it may be (is there a citation?) that 
running the Moses tokenizer before performing Byte-pair encoding (BPE) is preferable to either one or
the other. It is worth noting, however, that if you want to exactly specify your vocabulary size
at tokenization first, an exact-size tokenizer like BPE should be specified (and thus run) *last*. 


1. Moses ``tokenize.pl``: The classic, tried and true Perl tokenizer from the Moses decoder.
                          included here in the ``scripts/`` directory. Invoked with tokenier
                          type ``moses``.

2. Byte-Pair Encoding:    A compression-inspired unsupervised sub-word unit encoding
                          that performs well (Sennrich, 2016) and permits specification
                          of an exact vocabulary size. Native to ``xnmt``; written in Python.
                          Invoked with tokenizer type ``bpe``. 
                          *Specify which files from which the encoding is determined?*

3. [GOOGLETRONTHING]:     An extenral tokenizer library that permits a large number of tokenization
                          options, is written in C++, and is very fast. However, it must be installed
                          separately to ``xnmt``. 
                          Specification of the training file is set through the experiment framework,
                          but that (and all other) options can be passed transparently by adding them
                          to the experiment config.

4. External Tokenizers:   Any external tokenizer can be used as long as it tokenizes 
