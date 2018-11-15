.. _sec-writing-classes:

Visualization
=============

XNMT comes with several visualization tools.


Visualization of training progress
----------------------------------

The training progress can be monitored via Tensorboard. XNMT uses the ``tensorboardX`` package to write logs that can
be read and visualized via Tensorboard. These logs are written out by default, no configuration is required.
To run Tensorboard, Tensorflow must be installed first (see Tensorflow home page for further instructions):

.. code-block:: bash

  pip install tensorflow
  tensorboard --logdir <path/to/base/xnmt/log/dir>


Visualization of translation outputs
------------------------------------

Translation outputs can be analyzed via reporters as defined in ``xnmt/reports.py``. To use reporters, just simply define any reporter class inside the inference. 
Reports can only be used for inference-only experiments, i.e. experiments that load a pretrained model and only perform inference but no training.
The following reporters are available (see API doc for more details):

* ``AttentionReporter``: print attention matrices
* ``ReferenceDiffReporter``: HTML-visualization of diffs between reference and actual output
* ``CompareMtReporter``: perform detailed analysis, including computing over- and undergenerated n-grams.
* ``OOVStatisticsReporter``: compute OOV statistics, useful when using character- or subword models.
* ``SegmentationReporter``: Used only for the SegmentationSeqTransducer encoder, to print the segmentation of the input.
