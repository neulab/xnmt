#!/usr/bin/env python3

"""Usage: rehash_model.py MOD_IN MOD_OUT

Takes a trained model file with multiple saved checkpoints and converts these
checkpoints into standalone models.  This allows the different checkpoints to be
used, e.g., as parts of a model ensemble.

This script will:
- Analyze MOD_IN to find all saved model components
- Rehash all model components
- Write out the rehashed model to MOD_OUT

Options:
  -h, --help               Display this helpful text.
"""

from docopt import docopt
from glob import glob
import os
import shutil
import random

SUBCOL_TAG = "xnmt_subcol_name: "
SEP = "C"

subcol_rand = random.Random()

def generate_hash():
  rand_bits = subcol_rand.getrandbits(32)
  rand_hex = "%008x" % rand_bits
  return rand_hex

def extract_components(mod_infile):
  components = set([])
  yaml_lines = []
  with open(mod_infile, 'r', encoding="utf-8") as file_:
    for line in file_:
      yaml_lines.append(line)
      if SUBCOL_TAG in line:
        type_name, hash_name = line.strip().split(SUBCOL_TAG)[-1].split('.')
        components.add((type_name, hash_name))
  return components, yaml_lines

def rewrite_components(mod_infile, mod_outfile, components, yaml_lines):
  new_hashes = {} 
  os.mkdir(f"{mod_outfile}.data")
  for type_name, hash_name in components:
    new_hash = generate_hash()
    new_hashes[hash_name] = new_hash
    shutil.copy(f"{mod_infile}.data/{type_name}.{hash_name}",
                f"{mod_outfile}.data/{type_name}.{new_hash}")
  with open(f"{mod_infile}", 'r', encoding="utf-8") as file_in, \
       open(f"{mod_outfile}", 'w', encoding="utf-8") as file_out:
    for line in yaml_lines:
      if SUBCOL_TAG in line:
        part = line.strip().split('.')
        if part[-1] in new_hashes:
          line = line.replace(part[-1], new_hashes[part[-1]])
      file_out.write(line)

def main(mod_infile, mod_outfile):
  components, yaml_lines = extract_components(mod_infile)
  print("Found {:3d} model components.".format(len(components)))

  rewrite_components(mod_infile, mod_outfile, components, yaml_lines)

  print("All done.")


if __name__ == "__main__":
  args = docopt(__doc__)
  main(args['MOD_IN'], args['MOD_OUT'])
