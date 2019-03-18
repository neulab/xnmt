.. _save-file-format:

Switchable backends
===================

Overview
--------

*xnmt* supports switching between DyNet and Pytorch backends via command line arguments ``--backend dynet`` (default) or
``--backend torch``. Below we describe how this works and what to consider when implementing code.

Below are a few design principles:

* Executions with DyNet backend run without Pytorch installed, and vice versa.
* Configuration YAML files should be identical for both backends.
* High-level classes should work with either toolkit. Low-level classes usually require separate implementations for
  each backend. A rule-of-thumb is that features described with mathematical equations should be considered low-level
  and be implemented in a backend-dependent fashion, everything else as high-level with backend-independent code.
* It is not realistic to keep all features in sync between both backends. However, basic functionality should be
  maintained in parallel.
* Class and argument names should refer to the same concepts in both backends. For example, an argument named
  ``dropout`` in an RNN model could refer to various things, and one should make sure that it refers to the same thing
  with both backends.
* To the extent possible, running configuration files should either return (almost) identical results in both backends
  (in case all configured features are supported by both), or return a ``unsupported backend`` error message (in case
  some features are not implemented in one toolkit). It may not be practical to be overly strict about this, but it
  would be desirable to the extent possible.

Writing backend-specific code
-----------------------------

The backend is determined first thing during start up. It can be queried in the code using various mechanisms:

* ``xnmt.backend_dynet`` or ``xnmt.backend_torch`` boolean variables. This is for example used for conditionally
  importing backend-specific packages.
* ``@xnmt.require_dynet`` or ``@xnmt.require_torch`` class decorators. For implementing backend-specific versions of
  classes, per convention one would implement two classes  ``@xnmt.require_dynet class MyClassDynet:`` and
  ``@xnmt.require_torch class MyClassTorch:``. The implementation to be used is then selected by writing
  ``MyClass = xnmt.resolve_backend(MyClassDynet, MyClassTorch). Note that both classes should define the same
  ``yaml_tag=!MyClass`` in case they are ``Serializable``.
* In case a class is only implemented with one backend, this simplifies to e.g.
  ``@xnmt.require_dynet class MyClass:``, i.e. the class can receive its final name directly and ``resolve_backend``
  is not needed.

For a code example that uses all of the described concepts, refer to e.g. ``xnmt/transducers/recurrent.py``.

Writing backend-agnostic code
-----------------------------

*xnmt* provides a wrapper around the high-level features of both backends that deal with querying tensor dimensions,
concatenating or aggregating results, etc. This is purposefully not a complete wrapper and does not contain anything
that is considered low-level/math according to our definition, but is meant to enable writing backend-independent
high-level code. The wrapper is implemented under ``xnmt/tensor_tools.py``. An example where it is used is
``AutoRegressiveDecoder`` and ``DefaultTranslator``.