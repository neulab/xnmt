API Doc
=======

Experiment
----------

.. autoclass:: xnmt.experiment.Experiment
   :members:
   :show-inheritance:


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
   
.. autoclass:: xnmt.decoder.MlpSoftmaxDecoderState
   :members:
   :show-inheritance:
   

Inference
--------------

.. autoclass:: xnmt.inference.SimpleInference 
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

LengthNormalization
-------------------

.. autoclass:: xnmt.length_normalization.LengthNormalization
   :members:
   :show-inheritance:

.. autoclass:: xnmt.length_normalization.NoNormalization
   :members:
   :show-inheritance:

.. autoclass:: xnmt.length_normalization.AdditiveNormalization
   :members:
   :show-inheritance:

.. autoclass:: xnmt.length_normalization.PolynomialNormalization
   :members:
   :show-inheritance:

.. autoclass:: xnmt.length_normalization.MultinomialNormalization
   :members:
   :show-inheritance:

.. autoclass:: xnmt.length_normalization.GaussianNormalization
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

Vocab
-----

.. autoclass:: xnmt.vocab.Vocab
   :members:
   :show-inheritance:

Batcher
-----------

.. autoclass:: xnmt.batcher.Batch
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.Mask
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.Batcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.InOrderBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.SrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.TrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.SrcTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.TrgSrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.SentShuffleBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.WordShuffleBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.WordSrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.WordTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.WordSrcTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: xnmt.batcher.WordTrgSrcBatcher
   :members:
   :show-inheritance:

LossBuilder
--------------

.. autoclass:: xnmt.loss.LossBuilder
   :members:
   :show-inheritance:

.. autoclass:: xnmt.loss.LossScalarBuilder
   :members:
   :show-inheritance:

LossCalculator
--------------

.. autoclass:: xnmt.loss_calculator.LossCalculator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.loss_calculator.MLELoss
   :members:
   :show-inheritance:

.. autoclass:: xnmt.loss_calculator.ReinforceLoss
   :members:
   :show-inheritance:

Serialize
---------

.. autoclass:: xnmt.serialize.serializable.Serializable
   :show-inheritance:
   :members:

.. autofunction:: xnmt.serialize.serializable.bare

.. autoclass:: xnmt.serialize.tree_tools.Path
   :show-inheritance:
   :members:

.. autoclass:: xnmt.serialize.tree_tools.Ref
   :show-inheritance:
   :members:

Reportable
----------

.. autoclass:: xnmt.reports.Reportable
   :show-inheritance:
   :members:

GeneratorModel
--------------

.. autoclass:: xnmt.generator.GeneratorModel
   :show-inheritance:
   :members:



Other Classes
-------------

TODO: Add documentation.
