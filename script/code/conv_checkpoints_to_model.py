#!/usr/bin/env python3

"""Usage: conv_checkpoints_to_model.py MODFILE

Takes a trained model file with multiple saved checkpoints and converts these
checkpoints into standalone models.  This allows the different checkpoints to be
used, e.g., as parts of a model ensemble.

This script will:
- Analyze MODFILE to find all saved model components
- Rename all model components in the checkpoint directories
  (MODFILE.data.<checkpoint>/) to unique names
- Rename MODFILE.data.<checkpoint>/ to MODFILE.<checkpoint>.data/
- Create new standalone model files MODFILE.<checkpoint>

Options:
  -h, --help               Display this helpful text.
"""

from docopt import docopt
from glob import glob
import os
import sys

SUBCOL_TAG = "xnmt_subcol_name: "
SEP = "C"

def extract_components(modfile):
  components = set([])
  yaml_lines = []
  with open(modfile, 'r', encoding="utf-8") as file_:
    for line in file_:
      yaml_lines.append(line)
      if SUBCOL_TAG in line:
        components.add(line.strip().split(SUBCOL_TAG)[-1])
  return components, yaml_lines

def get_checkpoints(modfile, components):
  checkpoints = []
  for dirname in glob(f"{modfile}.data.*"):
    name = dirname.split(".")[-1]
    for part in components:
      assert os.path.exists("/".join((dirname, part))), \
        f"Model checkpoint {name} is missing {part}"
    checkpoints.append(name)
  return checkpoints

def rewrite_components(modfile, name, components, yaml_lines):
  for part in components:
    os.rename(f"{modfile}.data.{name}/{part}",
              f"{modfile}.data.{name}/{part}{SEP}{name}")
  os.rename(f"{modfile}.data.{name}",
            f"{modfile}.{name}.data")
  with open(f"{modfile}.{name}", 'w', encoding="utf-8") as file_:
    for line in yaml_lines:
      if SUBCOL_TAG in line:
        part = line.strip().split(SUBCOL_TAG)[-1]
        line = line.replace(f"{SUBCOL_TAG}{part}", f"{SUBCOL_TAG}{part}{SEP}{name}")
      file_.write(line)

def main(modfile):
  components, yaml_lines = extract_components(modfile)
  print("Found {:3d} model components.".format(len(components)))

  checkpoints = get_checkpoints(modfile, components)
  print("Found {:3d} extra model checkpoints.".format(len(checkpoints)))

  for name in checkpoints:
    print("Processing checkpoint: {:3s}".format(name))
    rewrite_components(modfile, name, components, yaml_lines)

  print("All done.")


if __name__ == "__main__":
  args = docopt(__doc__)
  main(args['MODFILE'])
