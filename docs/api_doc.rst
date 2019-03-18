.. _sec-api-doc:

API Doc
=======

The *xnmt* interface is documented below, with features not supported by the selected backend marked accordingly.

Experiment
----------

.. _mod-experiments:
.. automodule:: xnmt.experiments
   :members:
   :show-inheritance:

Model
-----

Model Base Classes
~~~~~~~~~~~~~~~~~~

.. automodule:: xnmt.models.base
   :members:
   :show-inheritance:

Translator
~~~~~~~~~~

.. automodule:: xnmt.models.translators
   :members:
   :show-inheritance:

Embedder
~~~~~~~~

.. automodule:: xnmt.modelparts.embedders
   :members:
   :show-inheritance:

Transducer
~~~~~~~~~~

.. automodule:: xnmt.transducers.base
   :members:
   :show-inheritance:

RNN
~~~

.. automodule:: xnmt.transducers.recurrent
   :members:
   :show-inheritance:

.. automodule:: xnmt.transducers.pyramidal
   :members:
   :show-inheritance:

.. automodule:: xnmt.transducers.residual
   :members:
   :show-inheritance:

Attender
~~~~~~~~

.. automodule:: xnmt.modelparts.attenders
   :members:
   :show-inheritance:


Decoder
~~~~~~~

.. automodule:: xnmt.modelparts.decoders
   :members:
   :show-inheritance:


Bridge
~~~~~~
.. automodule:: xnmt.modelparts.bridges
   :members:
   :show-inheritance:

Transform
~~~~~~~~~
.. automodule:: xnmt.modelparts.transforms
   :members:
   :show-inheritance:

Scorer
~~~~~~
.. automodule:: xnmt.modelparts.scorers
   :members:
   :show-inheritance:

SequenceLabeler
~~~~~~~~~~~~~~~
.. automodule:: xnmt.models.sequence_labelers
   :members:
   :show-inheritance:

Classifier
~~~~~~~~~~
.. automodule:: xnmt.models.classifiers
   :members:
   :show-inheritance:



Loss
----

Loss
~~~~

.. automodule:: xnmt.losses
   :members:
   :show-inheritance:

LossCalculator
~~~~~~~~~~~~~~

.. automodule:: xnmt.loss_calculators
   :members:
   :show-inheritance:

Training
--------

TrainingRegimen
~~~~~~~~~~~~~~~
.. automodule:: xnmt.train.regimens
   :members:
   :show-inheritance:

TrainingTask
~~~~~~~~~~~~
.. automodule:: xnmt.train.tasks
   :members:
   :show-inheritance:

Parameters
----------

ParamManager
~~~~~~~~~~~~
.. automodule:: xnmt.param_collections
   :members:
   :show-inheritance:


Optimizer
~~~~~~~~~

.. automodule:: xnmt.optimizers
   :members:
   :show-inheritance:

ParamInitializer
~~~~~~~~~~~~~~~~
.. automodule:: xnmt.param_initializers
   :members:
   :show-inheritance:


Inference
---------

AutoRegressiveInference
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xnmt.inferences
   :members:
   :show-inheritance:

SearchStrategy
~~~~~~~~~~~~~~

.. automodule:: xnmt.search_strategies
   :members:
   :show-inheritance:

LengthNormalization
~~~~~~~~~~~~~~~~~~~

.. automodule:: xnmt.length_norm
   :members:
   :show-inheritance:

Evaluation
----------

EvalTasks
~~~~~~~~~
.. automodule:: xnmt.eval.tasks
   :members:
   :show-inheritance:

Eval Metrics
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: xnmt.eval.metrics
   :members:
   :show-inheritance:



Data
----

Sentence
~~~~~~~~

.. automodule:: xnmt.sent
   :members:
   :show-inheritance:

InputReader
~~~~~~~~~~~

.. automodule:: xnmt.input_readers
   :members:
   :show-inheritance:

Vocab
~~~~~

.. automodule:: xnmt.vocabs
   :members:
   :show-inheritance:

Batcher
~~~~~~~

.. automodule:: xnmt.batchers
   :members:
   :show-inheritance:


Preprocessing
~~~~~~~~~~~~~

.. automodule:: xnmt.preproc
   :members:
   :show-inheritance:

Persistence
-------------

.. automodule:: xnmt.persistence
   :members:
   :show-inheritance:

Reportable
----------

.. automodule:: xnmt.reports
   :members:
   :show-inheritance:

Settings
--------

.. automodule:: xnmt.settings
   :members:
   :show-inheritance:
