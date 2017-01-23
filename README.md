eXtensible Neural Machine Translation
=====================================

This is a stub of a repository to create an extensible neural machine translation toolkit.
It is coded in Python based on [DyNet](http://github.com/clab/dynet).

Coding Style
------------

In general, follow Python conventions. Must be Python3 compatible. Functions should be snake case.

Try to name classes the same thing as the file name. File names should be hyphens, class names upper cased. For organization purposes, let's make the parent's name come first, so an `Evaluator` that measures BLEU should be `EvaluatorBleu`.

Indentation should be two whitespace characters.

Aim to write unit tests for most functionality where possible.
