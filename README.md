eXtensible Neural Machine Translation
=====================================

This is a repository for the extensible neural machine translation toolkit `xnmt`.
The over-arching goal of `xnmt` is that it be easy to use for research, and thus it supports
a modular design that means that new methods should be easy to implement by adding new modules.
It is coded in Python based on [DyNet](http://github.com/clab/dynet).

More information can be found in the [documentation](http://xnmt.readthedocs.io).

If you use `xnmt` in a research paper, we'd appreciate if you cite the following system description:

    @inproceedings{neubig18xnmt,
        title = {{XNMT}: The eXtensible Neural Machine Translation Toolkit},
        author = {Graham Neubig and Matthias Sperber and Xinyi Wang and Matthieu Felix and Austin Matthews and Sarguna Padmanabhan and Ye Qi and Devendra Singh Sachan and Philip Arthur and Pierre Godard and John Hewitt and Rachid Riad and Liming Wang },
        booktitle = {Conference of the Association for Machine Translation in the Americas (AMTA) Open Source Software Showcase},
        address = {Boston},
        month = {March},
        year = {2018}
    }

[![Build Status](https://travis-ci.org/neulab/xnmt.svg?branch=master)](https://travis-ci.org/neulab/xnmt)
[![Documentation Status](http://readthedocs.org/projects/xnmt/badge/?version=latest)](http://xnmt.readthedocs.io/en/latest/?badge=latest)
