from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Build import cythonize
import sys
import io
import os

sys.path.append("./xnmt")

with io.open("requirements.txt", encoding="utf-8") as req_fp:
  install_requires = req_fp.readlines()

extensions = Extension('xnmt.cython.xnmt_cython',
                       sources=['xnmt/cython/xnmt_cython.pyx',
                                'xnmt/cython/src/functions.cpp'],
                       language='c++',
                       extra_compile_args=["-std=c++11"],
                       extra_link_args=["-std=c++11"])


setup(
  name='xnmt',
  version='0.0.1',
  description='eXtensible Neural Machine Translation',
  author='neulab',
  license='Apache License',
  install_requires=install_requires,
  packages=[
      'xnmt',
  ],
  ext_modules=cythonize(extensions),
)
