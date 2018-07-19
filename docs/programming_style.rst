.. _sec-programming-style:

Programming Conventions
=======================

Philosphy
---------

The over-arching goal of *xnmt* is that it be easy to use for research. When implementing a new
method, it should require only minimal changes (e.g. ideally the changes will be limited to a
single file, over-riding an existing class). Obviously this ideal will not be realizable all the
time, but when designing new functionality, try to think of this goal. If there are tradeoffs,
the following is the order of priority (of course getting all is great!):

1. Code Correctness
2. Extensibility and Readability
3. Accuracy and Effectiveness of the Models
4. Efficiency

Style
-----

There are some minimal coding style conventions:

- Functions should be snake_case, classes should be UpperCamelCase.
- Indentation should be two whitespace characters.
- In variable names, common words should be abbreviated as:

  - source -> src
  - target -> trg
  - sentence -> sent
  - hypothesis -> hyp
  - reference -> ref


Documentation
-------------

- Docstrings should be made according to the `Google style guide <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/>`_.
- Types should be annotated consistently, see `corresponding Python docs <https://docs.python.org/3/library/typing.html>`_.
- Ideally, documentation should be added at module-level (giving a summary of the most relevant contents of the module),
  the class level (including arguments for ``__init__()``), and method level. Documentation for methods/classes etc.
  that do not need to be accessed from outside may be omitted and these should ideally marked as private by adding a
  single underscore as prefix.
- Note: some of these conventions are currently not followed consistently; PRs welcome!

Testing
-------

A collection of unit tests exists to make sure things don't break. When writing new code:

- The minimum recommendation is to add a config file to ``test/config`` and add a corresponding entry to
  ``test/test_run.py`` which will ensure that future commits will not cause this to crash. This "crash test config"
  should run as fast as possible.
- Even better would be correctness tests, several examples for which can be found in the test package.


Logging
-------

For printing output in a consistent and controllable way, a few conventions 
should be followed (see _official documentation: https://docs.python.org/3/howto/logging.html#when-to-use-logging for more details):

- ``logger.info()`` should be used for most outputs. Such outputs are assumed to be usually shown but can be turned off if needed.
- ``print()`` for regular output without which the execution would be incomplete. The main use case is to print final results, etc.
- ``logger.debug()`` for detailed information that isn't needed in normal operation
- ``logger.warning()``, logger.error() or logger.critical() for problematic situations
- ``yaml_logger(dict)`` for structured logging of information that should be easily automatically parseable and might be too bulky to print to the console.

These loggers can be requested as follows:

.. code-block:: python

  from xnmt import logger
  from xnmt import yaml_logger

Contributing
------------

Go ahead and send a pull request! If you're not sure whether something will be useful and
want to ask beforehand, feel free to open an issue on the github.
