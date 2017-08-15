Getting Started
===============

Prerequisites
-------------

Before running ``xnmt`` you must install the Python bindings for
`DyNet <http://github.com/clab/dynet>`_.

Training/testing a Model
------------------------

If you want to try to run a simple experiment, you can do so using sample 
configurations in the ``examples`` directory. For example, if you wnat to try
the default configuration file, which trains an attentional encoder-decoder model,
you can do so by running::

    python xnmt/xnmt_run_experiments.py examples/standard.yaml

The various examples that you can use are:

- ``examples/standard.yaml``: A standard neural MT model
- ``examples/speech.yaml``: An example of speech-to-text translation
- ``examples/debug.yaml``: A simple debugging configuration that should run super-fast
- ``examples/preproc.yaml``: A configuration including preprocessing directives like tokenization and filtering.

See ``experiments.md`` for more details about writing experiment configuration files
that allow you to specify the various 

Running unit tests
------------------

From the main directory, run: ``python -m unittest discover``

