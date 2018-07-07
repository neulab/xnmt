.. _save-file-format:

Save File Format
================

Overview
--------

When saving a (partly) trained model to disk, the resulting model file is in YAML format and looks very similar to the
configuration files (see :doc:`experiment_config_files`) with a few exceptions:

* Saved model files hold only one experiment (in contrast, config files contain dictionaries of several named
  experiments).
* Saved models are accompanied by a ``.data`` directory holding trained DyNet weights.
* Some components replace the originally specified arguments with updated contents. For instance, the vocabulary is
  usually stored as an explicit list in saved model files, whereas config files typically refer to an external vocab
  file.

.data sub-directory
-------------------
This directory contains a list of DyNet subcollections with names such as ``Linear.98dc700f`` or
``UniLSTMSeqTransducer.519cfb41``. Every ``Serializable`` class that allocates DyNet parameters using
``xnmt.param_collection.ParamManager.my_params(self)`` (see :doc:`writing_xnmt_classes`) will have one such
subcollection written to disk. The file names correspond to the component's ``xnmt_subcol_name``, consisting of the
component name and a unique identifier. The ``xnmt_subcol_name`` is also stored in the saved model's YAML file to
establish the correspondence. Each subcollection is stored using DyNet's serialization format which is a readable text
file.

In case several checkpoints are saved, there will be additional ``.data.1``, ``.data.2`` etc. files. It is worth
mentioning that ``xnmt_subcol_name`` does not change between checkpoints, and only one YAML file is written out. Also
note that the additional checkpoints are generally ignored when loading a saved model, but can be substituted manually
by renaming them, or be processed by the below utilities.

Command-line utilities
----------------------

* ``script/code/avg_checkpoints.py``: Perform checkpoint-averaging by taking the elementwise arithmetic average of
  parameters from all saved checkpoints.
* ``script/code/conv_checkpoints_to_model.py``: Convert a checkpoint to its own model. This is for example useful to
  enable checkpoint ensembling. Under the hood, this draw new random ``xnmt_subcol_name`` identifiers and in order to
  enable loading all checkpoints as separate models into XNMT.