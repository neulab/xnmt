.. xnmt documentation master file, created by
   sphinx-quickstart on Mon May 29 09:58:33 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

eXtensible Neural Machine Translation
=====================================

This is a repository for the extensible neural machine translation toolkit ``xnmt``.
It is coded in Python based on `DyNet <http://github.com/clab/dynet>`_.

Usage Directions
----------------

If you want to try to run an experiment, you can do so using sample configurations in the ``examples``
directory. For example, if you wnat to try the default configuration file,
which trains an attentional encoder-decoder model, you can do so by running

    python xnmt/xnmt_run_experiments.py examples/standard.yaml

There are other examples here:

- ``examples/standard.yaml``: A standard neural MT model
- ``examples/debug.yaml``: A simple debugging configuration that should run super-fast
- ``examples/speech.yaml``: An example of speech-to-text translation

See ``experiments.md`` for more details about writing experiment configuration files.

Programming Style
-----------------

The over-arching goal of ``xnmt`` is that it be easy to use for research. When implementing a new
method, it should require only minimal changes (e.g. ideally the changes will be limited to a
single file, over-riding an existing class). Obviously this ideal will not be realizable all the
time, but when designing new functionality, try to think of this goal.

There are also a minimal of coding style conventions:

- Follow Python conventions, and be Python2/3 compatible.
- Functions should be snake case.
- Indentation should be two whitespace characters.

We will aim to write unit tests to make sure things don't break, but these are not implemented yet.

In variable names, common words should be abbreviated as:

- source -> src
- target -> trg
- sentence -> sent
- hypothesis -> hyp


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
