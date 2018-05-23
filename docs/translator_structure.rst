.. _sec-translator-structure:

Translator Structure
====================

If you want to dig in to using *xnmt* for your research it is necessary to understand
the overall structure. The main class that you need to be aware of is ``Translator``, which
can calculate the conditional probability of the target sentence given the source sentence.
This is useful for calculating losses at training time, or generating sentences at test time.
Basically it consists of 5 major components:

1. Source ``Embedder``:       This converts input symbols into continuous-space vectors. Usually this
                              is done by looking up the word in a lookup table, but it could be done
                              any other way.
2. Encoder ``SeqTransducer``: Takes the embedded input and encodes it, for example using a bi-directional
                              LSTM to calculate context-sensitive embeddings.
3. ``Attender``:              This is the "attention" module, which takes the encoded input and decoder
                              state, then calculates attention.
4. Target ``Embedder``:       This converts output symbols into continuous-space vectors like its counterpart
                              in the source language.
5. ``Decoder``:               This calculates a probability distribution over the words in the output,
                              either to calculate a loss function during training, or to generate outputs
                              at test time.

In addition, given this ``Translator``, we have a ``SearchStrategy`` that takes the calculated
probabilities calculated by the decoder and actually generates outputs at test time.

There are a bunch of auxiliary classes as well to handle saving/loading of the inputs,
etc. However, if you're interested in using *xnmt* to develop a new method, most of your
work will probably go into one or a couple of the classes listed above.
