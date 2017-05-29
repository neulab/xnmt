eXtensible Neural Machine Translation
=====================================

This is a stub of a repository to create an extensible neural machine translation toolkit.
It is coded in Python based on [DyNet](http://github.com/clab/dynet).

Usage Directions
----------------

### Running experiments with `xnmt-run-experiments`

Configuration files are in [YAML dictionary format](https://docs.ansible.com/ansible/YAMLSyntax.html).

Top-level entries in the file correspond to individual experiments to run. Each
such entry must have four subsections: `experiment`, `train`, `decode`,
and `evaluate`. Options for each subsection are listed below.

There can be a special top-level entry named `defaults`; if it is
present, parameters defined in it will act as defaults for other experiments
in the configuration file.

The stdout and stderr outputs of an experiment will be written to `<experiment-name>.log`
and `<experiment-name>.err.log` in the current directory.

See [experiments.md](experiments.md) for option details, and [the default configuration
file](test/experiments-config.yaml) for an example configuration you can use.

Coding Style
------------

In general, follow Python conventions. Must be Python3 compatible. Functions should be snake case.

Indentation should be two whitespace characters.

Aim to write unit tests for most functionality where possible.

In variable names, common words should be abbreviated as:
* source -> src
* target -> trg
* sentence -> sent
* hypothesis -> hyp
