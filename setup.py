from setuptools import setup, find_packages
import sys
import io
import xnmt

with io.open("requirements.txt", encoding="utf-8") as req_fp:
  install_requires = req_fp.readlines()

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
)
