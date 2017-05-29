Translator Structure
====================

If you want to dig in to using ``xnmt`` for your research it is necessary to understand
the overall structure. Basically it consists of 5 major components:

1. **Embedder**: This converts input symbols into continuous-space vectors. Usually this
                 is done by looking up the word in a lookup table, but it could be done
                 any other way.
2. **Encoder**:  Takes the embedded input and encodes it, for example using a bi-directional
                 LSTM to calculate context-sensitive embeddings.
3. **Attender**: This is the "attention" module, which takes the encoded input and decoder
                 state, then calculates attention.
4. **Decoder**:  This calculates a probability distribution over the words in the output,
                 either to calculate a loss function during training, or to generate outputs
                 at test time.
5. **SearchStrategy**: This takes the probabilities calculated by the decoder and actually
                 generates outputs at test time.

There are a bunch of auxiliary classes as well to handle saving/loading of the inputs,
etc. However, if you're interested in using ``xnmt`` to develop a new method, most of your
work will probably go into one or a couple of the classes listed above.
