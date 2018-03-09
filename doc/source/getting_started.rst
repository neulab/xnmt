Getting Started
===============

Prerequisites
-------------

``xnmt`` requires Python 3.6.

Before running ``xnmt`` you must install the required packages, including Python bindings for
`DyNet <http://github.com/clab/dynet>`_.
This can be done by running ``pip -r requirements.txt``

Next, install ``xnmt`` by running ``python setup.py install`` for normal usage or ``python setup.py develop`` for development.

Running the examples
--------------------

``xnmt`` includes a series of tutorial-style examples in the examples subfolder.
These are a good starting point to get familiarized with specifying models and
experiments. To run the first experiment, use the following::

    python -m xnmt.xnmt_run_experiments examples/01_standard.yaml

Make sure to read the comments provided in ``examples/01_standard.yaml``.

See ``experiments.md`` for more details about writing experiment configuration files
that allow you to specify the various 

Running unit tests
------------------

From the main directory, run: ``python -m unittest discover``
Or, to run a specific test, use e.g. ``python -m unittest test.test_run.TestRunningConfig.test_standard``

Cython modules
------------------

If you wish to use all the modules in xnmt that need cython, you need to build the cython extensions by this command::

  python setup.py build_ext --inplace --use-cython-extensions
