#!/usr/bin/env python3

"""
Reads experiments descriptions in the passed configuration file
and set up models in a server that can repeatedly be queried for inference on files.

Run like so
python3 xnmt_run_server.py <experiment filename>.yaml --<various options>.

This launches a server at http://localhost:PORT where PORT is specified in the
file.

Then send POST or GET requests with the 'q' key's value set to whatever the text
of the minified/decompiled/src code text is. This server will the send back
the output translation the form '<Line number> ||| <Translated text>' in order
by line number, with the line number starting from 0 and corresponding to the
line number of the source text.
"""

import logging
logger = logging.getLogger('xnmt')

import argparse
import sys
import random
import numpy as np
import web #easy_install web.py
import json
import os
import inspect

if not any(a.startswith("--settings") for a in sys.argv): sys.argv.insert(1, "--settings=settings.standard")
from simple_settings import settings
if settings.RESOURCE_WARNINGS:
  import warnings
  warnings.simplefilter('always', ResourceWarning)

from xnmt.serialize.options import OptionParser
from xnmt.tee import Tee
from xnmt.serialize.serializer import YamlSerializer


web.config.debug = False
PORT = 9093

model_list = None
urls = ('/evaluate', 'evaluate')
TMP_SRC_FILE_PATH = 'xnmt_server_current_src.tmp'
TMP_TRG_FILE_PATH = 'xnmt_server_current_trg.tmp'


class MyApplication(web.application): 
  def run(self, port=8080, *middleware): 
      func = self.wsgifunc(*middleware)
      return web.httpserver.runsimple(func, ('0.0.0.0', port))

class evaluate:
  def GET(self):
    i = web.input(_unicode=False, q="no data")
    output = predict(model_list, i)
    return output   

  def POST(self):
    return self.GET()

app = MyApplication(urls, globals())


'''
Assume that the entirety of the text is stored under the key 'q'.
'''
def predict(exp_list, text):
  src_text = text.q
  result = ""

  with open(TMP_SRC_FILE_PATH, 'w', encoding='utf-8') as out_file:
    out_file.write(text.q)
  
  for exp in exp_list:
    exp.model.inference(exp.model, src_file=TMP_SRC_FILE_PATH, trg_file=TMP_TRG_FILE_PATH)
    with open(TMP_TRG_FILE_PATH, 'r', encoding='utf8') as in_file:
      result = in_file.readlines()
      result = "".join(list(map(lambda x: str(x[0]) + ' ||| ' + x[1], enumerate(result))))
    os.remove(TMP_SRC_FILE_PATH)
    os.remove(TMP_TRG_FILE_PATH)

  logger.info(f"Translating text of length {len(src_text)}.")

  return result


def setupServer(args):
  config_parser = OptionParser()

  if args.dynet_seed:
    random.seed(args.dynet_seed)
    np.random.seed(args.dynet_seed)

  import xnmt.serialize.imports
  config_experiment_names = config_parser.experiment_names_from_file(args.experiments_file)

  experiment_list = []
  experiment_names = args.experiment_name or config_experiment_names

  if args.experiment_name:
    nonexistent = set(experiment_names).difference(config_experiment_names)
    if len(nonexistent) != 0:
      raise Exception("Experiments {} do not exist".format(",".join(list(nonexistent))))

  for experiment_name in experiment_names:
    '''
    Loads each experiment into memory, but does not run any of the training
    or evaluation.
    '''

    uninitialized_exp_args = config_parser.parse_experiment(args.experiments_file, experiment_name)
    yaml_serializer = YamlSerializer()
    uninitialized_exp_args.data.exp_global.commandline_args = args
    experiment = yaml_serializer.initialize_if_needed(uninitialized_exp_args)
    experiment_list.append(experiment)
    logger.info(f"Loaded models from experiment {experiment_name}.")

  return experiment_list


def parse_arguments(overwrite_args=None):
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--dynet-mem", type=int)
  argparser.add_argument("--dynet-seed", type=int)
  argparser.add_argument("--dynet-autobatch", type=int)
  argparser.add_argument("--dynet-devices", type=str)
  argparser.add_argument("--dynet-viz", action='store_true', help="use visualization")
  argparser.add_argument("--dynet-gpu", action='store_true', help="use GPU acceleration")
  argparser.add_argument("--dynet-gpu-ids", type=int)
  argparser.add_argument("--dynet-gpus", type=int)
  argparser.add_argument("--dynet-weight-decay", type=float)
  argparser.add_argument("--dynet-profiling", type=int)
  argparser.add_argument("--settings", type=str, default="standard")
  argparser.add_argument("experiments_file")
  argparser.add_argument("experiment_name", nargs='*', help="Run only the specified experiments")
  argparser.set_defaults(generate_doc=False)
  return argparser.parse_args(overwrite_args)


if __name__ == '__main__':
  import _dynet
  dyparams = _dynet.DynetParams()
  dyparams.from_args()
  args = parse_arguments()
  model_list = setupServer(args)
  logger.info(f"Running dynet server on port {PORT}.")
  app.run(port=PORT)

