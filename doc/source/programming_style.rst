
Programming Style
=================

Philosphy
---------

The over-arching goal of ``xnmt`` is that it be easy to use for research. When implementing a new
method, it should require only minimal changes (e.g. ideally the changes will be limited to a
single file, over-riding an existing class). Obviously this ideal will not be realizable all the
time, but when designing new functionality, try to think of this goal. If there are tradeoffs,
the following is the order of priority (of course getting all is great!):

1. Code Correctness
2. Extensibility and Readability
3. Accuracy and Effectiveness of the Models
4. Efficiency

Coding Conventions
------------------

There are also a minimal of coding style conventions:

- Follow Python 3 conventions, Python 2 is no longer supported.
- Functions should be snake_case, classes should be UpperCamelCase.
- Indentation should be two whitespace characters.
- Docstrings should be made in reST format (e.g. ``:param param_name:``, ``:returns:`` etc.)

A collection of unit tests exists to make sure things don't break.

In variable names, common words should be abbreviated as:

- source -> src
- target -> trg
- sentence -> sent
- hypothesis -> hyp
- reference -> ref

For printing output in a consistent and controllable way, a few conventions
should be followed (see _official documentation: https://docs.python.org/3/howto/logging.html#when-to-use-logging for more details):
- logger.info() should be used for most outputs. Such outputs are assumed to
  be usually shown but can be turned off if needed.
- print() for regular output without which the execution would be incomplete.
  The main use case is to print final results, etc.
- logger.debug() for detailed information that isn't needed in normal operation
- logger.warning(), logger.error() or logger.critical() for problematic situations
- yaml_logger(dict) for structured logging of information that should be easily
  automatically parseable and might be too bulky to print to the console.
These loggers can be requested as follows:

::
  import logging
  logger = logging.getLogger('xnmt')
  yaml_logger = logging.getLogger('yaml')

Contributing
------------

Go ahead and send a pull request! If you're not sure whether something will be useful and
want to ask beforehand, feel free to open an issue on the github.
