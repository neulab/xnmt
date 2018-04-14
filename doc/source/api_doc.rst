API Doc
=======

Experiment
----------

.. autoclass:: experiment.Experiment
   :members:
   :show-inheritance:

.. autoclass:: experiment.Experiment
   :members:
   :show-inheritance:

.. autoclass:: exp_global.ExpGlobal
   :members:
   :show-inheritance:

Model
-----

GeneratorModel
~~~~~~~~~~~~~~

.. autoclass:: generator.GeneratorModel
   :show-inheritance:
   :members:

Translator
~~~~~~~~~~

.. autoclass:: translator.Translator
   :members:

.. autoclass:: translator.DefaultTranslator
   :members:
   :show-inheritance:

.. autoclass:: translator.TransformerTranslator
   :members:
   :show-inheritance:

Embedder
~~~~~~~~

.. autoclass:: embedder.Embedder
   :members:

.. autoclass:: embedder.SimpleWordEmbedder
   :members:
   :show-inheritance:

.. autoclass:: embedder.NoopEmbedder
   :members:
   :show-inheritance:

Transducer
~~~~~~~~~~

.. autoclass:: transducer.Transducer
   :members:

.. autoclass:: transducer.SeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: transducer.FinalTransducerState
   :members:
   :show-inheritance:

.. autoclass:: transducer.ModularSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: transducer.IdentitySeqTransducer
   :members:
   :show-inheritance:

RNN
~~~

.. autoclass:: lstm.UniLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: lstm.BiLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: pyramidal.PyramidalLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: residual.ResidualLSTMSeqTransducer
   :members:
   :show-inheritance:

.. autoclass:: residual.ResidualRNNBuilder
   :members:
   :show-inheritance:

.. autoclass:: residual.ResidualBiRNNBuilder
   :members:
   :show-inheritance:

Attender
~~~~~~~~

.. autoclass:: attender.Attender
   :members:

.. autoclass:: attender.MlpAttender
   :members:
   :show-inheritance:

.. autoclass:: attender.DotAttender
   :members:
   :show-inheritance:

.. autoclass:: attender.BilinearAttender
   :members:
   :show-inheritance:

Decoder
~~~~~~~

.. autoclass:: decoder.Decoder
   :members:

.. autoclass:: decoder.MlpSoftmaxDecoder
   :members:
   :show-inheritance:

.. autoclass:: decoder.MlpSoftmaxDecoderState
   :members:
   :show-inheritance:

Bridge
~~~~~~
.. autoclass:: bridge.Bridge
   :members:
   :show-inheritance:

.. autoclass:: bridge.NoBridge
   :members:
   :show-inheritance:

.. autoclass:: bridge.CopyBridge
   :members:
   :show-inheritance:

.. autoclass:: bridge.LinearBridge
   :members:
   :show-inheritance:

Linear
~~~~~~
.. autoclass:: linear.Linear
   :members:
   :show-inheritance:

Loss
----

LossBuilder
~~~~~~~~~~~

.. autoclass:: loss.LossBuilder
   :members:
   :show-inheritance:

.. autoclass:: loss.LossScalarBuilder
   :members:
   :show-inheritance:

LossCalculator
~~~~~~~~~~~~~~

.. autoclass:: loss_calculator.LossCalculator
   :members:
   :show-inheritance:

.. autoclass:: loss_calculator.MLELoss
   :members:
   :show-inheritance:

.. autoclass:: loss_calculator.ReinforceLoss
   :members:
   :show-inheritance:

Training
--------

TrainingRegimen
~~~~~~~~~~~~~~~
.. autoclass:: training_regimen.TrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: training_regimen.SimpleTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: training_regimen.MultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: training_regimen.SameBatchMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: training_regimen.AlternatingBatchMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

.. autoclass:: training_regimen.SerialMultiTaskTrainingRegimen
   :members:
   :show-inheritance:

TrainingTask
~~~~~~~~~~~~
.. autoclass:: training_task.TrainingTask
   :members:
   :show-inheritance:

.. autoclass:: training_task.SimpleTrainingTask
   :members:
   :show-inheritance:

Parameters
----------

ParamManager
~~~~~~~~~~~~
.. autoclass:: param_collection.ParamManager
   :members:
   :show-inheritance:

ParamCollection
~~~~~~~~~~~~~~~
.. autoclass:: param_collection.ParamCollection
   :members:
   :show-inheritance:




Optimizer
~~~~~~~~~

.. autoclass:: optimizer.XnmtOptimizer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.SimpleSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.MomentumSGDTrainer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.AdagradTrainer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.AdadeltaTrainer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.AdamTrainer
   :members:
   :show-inheritance:

.. autoclass:: optimizer.TransformerAdamTrainer
   :members:
   :show-inheritance:

ParamInitializer
~~~~~~~~~~~~~~~~
.. autoclass:: param_init.ParamInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.NormalInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.UniformInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.ConstInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.GlorotInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.FromFileInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.NumpyInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.ZeroInitializer
   :members:
   :show-inheritance:

.. autoclass:: param_init.LeCunUniformInitializer
   :members:
   :show-inheritance:


Inference
---------

SimpleInference
~~~~~~~~~~~~~~~

.. autoclass:: inference.SimpleInference
   :members:
   :show-inheritance:

SearchStrategy
~~~~~~~~~~~~~~

.. autoclass:: search_strategy.SearchStrategy
   :members:
   :show-inheritance:

.. autoclass:: search_strategy.BeamSearch
   :members:
   :show-inheritance:

LengthNormalization
~~~~~~~~~~~~~~~~~~~

.. autoclass:: length_normalization.LengthNormalization
   :members:
   :show-inheritance:

.. autoclass:: length_normalization.NoNormalization
   :members:
   :show-inheritance:

.. autoclass:: length_normalization.AdditiveNormalization
   :members:
   :show-inheritance:

.. autoclass:: length_normalization.PolynomialNormalization
   :members:
   :show-inheritance:

.. autoclass:: length_normalization.MultinomialNormalization
   :members:
   :show-inheritance:

.. autoclass:: length_normalization.GaussianNormalization
   :members:
   :show-inheritance:

Evaluation
----------

EvalTaks
~~~~~~~~
.. autoclass:: eval_task.EvalTask
   :members:
   :show-inheritance:

.. autoclass:: eval_task.LossEvalTask
   :members:
   :show-inheritance:

.. autoclass:: eval_task.AccuracyEvalTask
   :members:
   :show-inheritance:

EvalScore
~~~~~~~~~

.. autoclass:: evaluator.EvalScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.LossScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.BLEUScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.GLEUScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.WERScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.CERScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.RecallScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.ExternalScore
   :members:
   :show-inheritance:

.. autoclass:: evaluator.SequenceAccuracyScore
   :members:
   :show-inheritance:


Evaluator
~~~~~~~~~
.. autoclass:: evaluator.Evaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.BLEUEvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.GLEUEvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.WEREvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.CEREvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.ExternalEvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.RecallEvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.MeanAvgPrecisionEvaluator
   :members:
   :show-inheritance:

.. autoclass:: evaluator.SequenceAccuracyEvaluator
   :members:
   :show-inheritance:



Data
----

Input
~~~~~

.. autoclass:: input.Input
   :members:
   :show-inheritance:

.. autoclass:: input.SimpleSentenceInput
   :members:
   :show-inheritance:

.. autoclass:: input.ArrayInput
   :members:
   :show-inheritance:

InputReader
~~~~~~~~~~~

.. autoclass:: input_reader.InputReader
   :members:
   :show-inheritance:

.. autoclass:: input_reader.BaseTextReader
   :members:
   :show-inheritance:

.. autoclass:: input_reader.PlainTextReader
   :members:
   :show-inheritance:

.. autoclass:: input_reader.SegmentationTextReader
   :members:
   :show-inheritance:

.. autoclass:: input_reader.ContVecReader
   :members:
   :show-inheritance:

.. autoclass:: input_reader.IDReader
   :members:
   :show-inheritance:

Vocab
~~~~~

.. autoclass:: vocab.Vocab
   :members:
   :show-inheritance:

Batcher
~~~~~~~

.. autoclass:: batcher.Batch
   :members:
   :show-inheritance:

.. autoclass:: batcher.Mask
   :members:
   :show-inheritance:

.. autoclass:: batcher.Batcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.InOrderBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.SrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.TrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.SrcTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.TrgSrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.SentShuffleBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.WordShuffleBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.WordSrcBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.WordTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.WordSrcTrgBatcher
   :members:
   :show-inheritance:

.. autoclass:: batcher.WordTrgSrcBatcher
   :members:
   :show-inheritance:

Preprocessing
~~~~~~~~~~~~~

.. autoclass:: preproc_runner.PreprocRunner
   :members:
   :show-inheritance:

Serialization
-------------

.. autoclass:: serialize.serializable.Serializable
   :show-inheritance:
   :members:

.. autofunction:: serialize.serializable.bare

.. autoclass:: serialize.tree_tools.Path
   :show-inheritance:
   :members:

.. autoclass:: serialize.tree_tools.Ref
   :show-inheritance:
   :members:

Reportable
----------

.. autoclass:: reports.Reportable
   :show-inheritance:
   :members:
