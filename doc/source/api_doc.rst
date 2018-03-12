API Doc
=======

Experiment
----------

.. autoclass:: xnmt.experiment.Experiment
   :members:
   :show-inheritance:

.. autoclass:: xnmt.exp_global.ExpGlobal
   :members:
   :show-inheritance:

Model
-----

GeneratorModel
~~~~~~~~~~~~~~

.. autoclass:: xnmt.generator.GeneratorModel
   :show-inheritance:
   :members:

Translator
~~~~~~~~~~

.. autoclass:: xnmt.translator.Translator
   :members:

.. autoclass:: xnmt.translator.DefaultTranslator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.translator.TransformerTranslator
   :members:
   :show-inheritance:

Embedder
~~~~~~~~

.. autoclass:: xnmt.embedder.Embedder
   :members:

.. autoclass:: xnmt.embedder.SimpleWordEmbedder
   :members:
   :show-inheritance:

.. autoclass:: xnmt.embedder.NoopEmbedder
   :members:
   :show-inheritance:

Transducer
~~~~~~~~~~

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
~~~

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
~~~~~~~~

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
~~~~~~~

.. autoclass:: xnmt.decoder.Decoder
   :members:

.. autoclass:: xnmt.decoder.MlpSoftmaxDecoder
   :members:
   :show-inheritance:
   
.. autoclass:: xnmt.decoder.MlpSoftmaxDecoderState
   :members:
   :show-inheritance:

Bridge
~~~~~~
.. autoclass:: xnmt.bridge.Bridge
   :members:
   :show-inheritance:

.. autoclass:: xnmt.bridge.NoBridge
   :members:
   :show-inheritance:

.. autoclass:: xnmt.bridge.CopyBridge
   :members:
   :show-inheritance:

.. autoclass:: xnmt.bridge.LinearBridge
   :members:
   :show-inheritance:

Linear
~~~~~~
.. autoclass:: xnmt.linear.Linear
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

Training
--------

TrainingRegimen
~~~~~~~~~~~~~~~
.. autoclass:: xnmt.training_regimen.TrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_regimen.SimpleTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_regimen.MultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_regimen.SameBatchMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_regimen.AlternatingBatchMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_regimen.SerialMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

TrainingTask
~~~~~~~~~~~~
.. autoclass:: xnmt.training_task.TrainingTask
   :members:
   :show-inheritance:

.. autoclass:: xnmt.training_task.SimpleTrainingTask
   :members:
   :show-inheritance:

Parameters
----------

PersistentParamCollection
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: xnmt.exp_global.PersistentParamCollection
   :members:
   :show-inheritance:


Optimizer
~~~~~~~~~

.. autoclass:: xnmt.optimizer.XnmtOptimizer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.SimpleSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.MomentumSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.AdagradTrainer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.AdadeltaTrainer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.AdamTrainer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.optimizer.TransformerAdamTrainer
   :members:
   :show-inheritance:

ParamInitializer
~~~~~~~~~~~~~~~~
.. autoclass:: xnmt.param_init.ParamInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.NormalInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.UniformInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.ConstInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.GlorotInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.FromFileInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.NumpyInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.ZeroInitializer
   :members:
   :show-inheritance:

.. autoclass:: xnmt.param_init.LeCunUniformInitializer
   :members:
   :show-inheritance:


Inference
---------

SimpleInference
~~~~~~~~~~~~~~~

.. autoclass:: xnmt.inference.SimpleInference 
   :members:
   :show-inheritance:

SearchStrategy
~~~~~~~~~~~~~~

.. autoclass:: xnmt.search_strategy.SearchStrategy 
   :members:
   :show-inheritance:

.. autoclass:: xnmt.search_strategy.BeamSearch
   :members:
   :show-inheritance:

LengthNormalization
~~~~~~~~~~~~~~~~~~~

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

Evaluation
----------

EvalTaks
~~~~~~~~
.. autoclass:: xnmt.eval_task.EvalTask
   :members:
   :show-inheritance:

.. autoclass:: xnmt.eval_task.LossEvalTask
   :members:
   :show-inheritance:

.. autoclass:: xnmt.eval_task.AccuracyEvalTask
   :members:
   :show-inheritance:

EvalScore
~~~~~~~~~

.. autoclass:: xnmt.evaluator.EvalScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.LossScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.BLEUScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.GLEUScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.WERScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.CERScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.RecallScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.ExternalScore
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.SequenceAccuracyScore
   :members:
   :show-inheritance:


Evaluator
~~~~~~~~~
.. autoclass:: xnmt.evaluator.Evaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.BLEUEvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.GLEUEvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.WEREvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.CEREvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.ExternalEvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.RecallEvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.MeanAvgPrecisionEvaluator
   :members:
   :show-inheritance:

.. autoclass:: xnmt.evaluator.SequenceAccuracyEvaluator
   :members:
   :show-inheritance:



Data
----

Input
~~~~~

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
~~~~~~~~~~~

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
~~~~~

.. autoclass:: xnmt.vocab.Vocab
   :members:
   :show-inheritance:

Batcher
~~~~~~~

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

Preprocessing
~~~~~~~~~~~~~

.. autoclass:: xnmt.preproc_runner.PreprocRunner
   :members:
   :show-inheritance:

Serialization
-------------

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
