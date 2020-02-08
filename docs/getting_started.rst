.. _sec-getting-started:

Getting Started
===============

Prerequisites
-------------

*xnmt* requires Python 3.6.

Before running *xnmt* you must install the required packages.
This can be done by running ``pip install -r requirements.txt``.
Depending on the backend, either Python bindings for `DyNet <http://github.com/clab/dynet>`_ or
`Pytorch <https://pytorch.org>`_ must be installed.
(There are also optional package requirements under ``requirements-extra/`` for features that are non-central to *xnmt*.)

Next, install *xnmt* by running ``python setup.py install`` for normal usage or ``python setup.py develop`` for
development.

Command line tools
------------------

*xnmt* comes with the following command line interfaces:

* ``xnmt`` runs experiments given a configuration file that can specify preprocessing, model training, and evaluation.
  The corresponding Python file is ``xnmt/xnmt_run_experiments.py``. Typical example call::

    xnmt --dynet-gpu my-training.yaml # for DyNet backend
    xnmt --backend torch --gpu my-training.yaml # for PyTorch backend

* ``xnmt_decode`` decodes a hypothesis using a specified model. The corresponding Python file is
  ``xnmt/xnmt_decode.py``. Typical example call::

    xnmt_decode --src src.txt --hyp out.txt --mod saved-model.mod

* ``xnmt_evaluate`` computes an evaluation metric given hypothesis and reference files. The corresponding Python file
  is ``xnmt/xnmt_evaluate.py``. Typical example call::

    xnmt_evaluate --hyp out.txt --ref ref.txt --metric bleu


Running the examples
--------------------

*xnmt* includes a series of tutorial-style examples in the ``examples/`` subfolder.
These are a good starting point to get familiarized with specifying models and
experiments. To run the first experiment, use the following::

    xnmt examples/01_standard.yaml

This is a shortcut for typing ``python -m xnmt.xnmt_run_experiments examples/01_standard.yaml``.
Make sure to read the comments provided in the :ref:`example configuration <ex-standard>`.

See the :ref:`sec-exp-conf` documentation entry for more details about writing experiment configuration files.

Switching Backends
------------------

*xnmt* has been developed based on DyNet, but can also be used with a Pytorch backend. Configuration files for both
are identical, the backend is switched by adding a command line argument ``--backend dynet`` (default) or
``--backend torch``. Note that Pytorch backend is currently experimental and supports a reduced number of features,
although most commonly models are supported with both backends.

Running recipes
---------------

*xnmt* includes several self-contained recipes on publically available data with competitive model settings, and
including scripts for data preparation, in the ``recipes/`` subfolder.

Running unit tests
------------------

From the main directory, run: ``python -m unittest``

Or, to run a specific test, use e.g. ``python -m unittest test.test_run.TestRunningConfig.test_standard``

Pytorch-supported unit tests are run by setting an environment variable: ``XNMT_BACKEND=torch python -m unittest``

Cython modules
------------------

If you wish to use all the modules in *xnmt* that need cython, you need to build the cython extensions by this command::

  python setup.py build_ext --inplace --use-cython-extensions
