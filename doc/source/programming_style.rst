
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

We will aim to write unit tests to make sure things don't break, but these are not implemented yet.

In variable names, common words should be abbreviated as:

- source -> src
- target -> trg
- sentence -> sent
- hypothesis -> hyp
- reference -> ref

Contributing
------------

Go ahead and send a pull request! If you're not sure whether something will be useful and
want to ask beforehand, feel free to open an issue on the github.
