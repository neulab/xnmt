API Doc
=======

Translator
----------

.. autoclass:: xnmt.translator.Translator
   :members:

.. autoclass:: xnmt.translator.DefaultTranslator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.translator.TransformerTranslator
   :members:
   :show-inheritance:

Embedder
--------

.. autoclass:: xnmt.embedder.Embedder
   :members:

.. autoclass:: xnmt.embedder.SimpleWordEmbedder
   :members:
   :show-inheritance:

.. autoclass:: xnmt.embedder.NoopEmbedder
   :members:
   :show-inheritance:

Transducer
----------

.. autoclass:: xnmt.transducer.Transducer
   :members:

.. autoclass:: xnmt.transducer.SeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.transducer.FinalTransducerState
   :members:
   :show-inheritance:

.. autoclass:: xnmt.transducer.ModularSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.transducer.IdentitySeqTransducer
   :members:
   :show-inheritance:

RNN
---

.. autoclass:: xnmt.lstm.UniLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.lstm.BiLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.pyramidal.PyramidalLSTMSeqTransducer
   :members:
   :show-inheritance:
   
.. autoclass:: xnmt.residual.ResidualLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.residual.ResidualRNNBuilder
   :members:
   :show-inheritance:

.. autoclass:: xnmt.residual.ResidualBiRNNBuilder
   :members:
   :show-inheritance:
   
Attender
--------

.. autoclass:: xnmt.attender.Attender
   :members:

.. autoclass:: xnmt.attender.MlpAttender
   :members:
   :show-inheritance:

.. autoclass:: xnmt.attender.DotAttender
   :members:
   :show-inheritance:

.. autoclass:: xnmt.attender.BilinearAttender
   :members:
   :show-inheritance:

Decoder
-------

.. autoclass:: xnmt.decoder.Decoder
   :members:

.. autoclass:: xnmt.decoder.MlpSoftmaxDecoder
   :members:
   :show-inheritance:

SearchStrategy
--------------

.. autoclass:: xnmt.search_strategy.SearchStrategy 
   :members:
   :show-inheritance:

.. autoclass:: xnmt.search_strategy.BeamSearch
   :members:
   :show-inheritance:

Input
-----

.. autoclass:: xnmt.input.Input 
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input.SimpleSentenceInput
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input.ArrayInput
   :members:
   :show-inheritance:

InputReader
-----------

.. autoclass:: xnmt.input_reader.InputReader
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input_reader.BaseTextReader
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input_reader.PlainTextReader
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input_reader.SegmentationTextReader
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input_reader.ContVecReader
   :members:
   :show-inheritance:

.. autoclass:: xnmt.input_reader.IDReader
   :members:
   :show-inheritance:



Other Classes
-------------

TODO: Add documentation.
