import argparse, sys, os, shutil
import os.path

import dynet as dy

def avg_subcol(data_files, out_file, subcol_name):
  load_param_col = dy.Model()
  save_param_col = dy.Model()
  save_param_subcol = save_param_col.add_subcollection(subcol_name)

  param_index = 0
  while True:
    is_lookup = False
    try:
      loaded_params = [load_param_col.load_param(df, f"/{subcol_name}/_{param_index}") for df in data_files]
    except RuntimeError:
      try:
        loaded_params = [load_param_col.load_lookup_param(df, f"/{subcol_name}/_{param_index}") for df in data_files]
        is_lookup = True
      except RuntimeError:
        break
    print(f"averaging /{subcol_name}/_{param_index}..")
    arrs = [p.as_array() for p in loaded_params]

    avg_arr = None
    for arr in arrs:
      if avg_arr is not None: avg_arr += arr
      else: avg_arr = arr
    avg_arr /= len(arrs)

    if is_lookup:
      save_param_subcol.lookup_parameters_from_numpy(avg_arr)
    else:
      save_param_subcol.parameters_from_numpy(avg_arr)

    param_index += 1

  save_param_subcol.save(out_file)

def avg_checkpoints(data_dirs, out_dir):
  os.makedirs(out_dir)
  for subcol_name in os.listdir(data_dirs[0]):
    avg_subcol([os.path.join(dd, subcol_name) for dd in data_dirs],
               os.path.join(out_dir, subcol_name),
               subcol_name)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_from", help="Path to read model file from")
  parser.add_argument("model_to", help="Path to write averaged model file to")
  args = parser.parse_args()

  if not os.path.isfile(args.model_from):
    raise ValueError(f"model_from does not exist: {args.model_from}")
  if os.path.isfile(args.model_to):
    raise ValueError(f"model_to already exists: {args.model_to}")
  elif not os.path.isdir(f"{args.model_from}.data") or not os.path.isdir(f"{args.model_from}.data.1"):
    raise ValueError("avg_checkpoints can only be applied to models saved with at least 2 checkpoints")

  data_files = [f"{args.model_from}.data"]
  i=1
  while True:
    data_path = f"{args.model_from}.data.{i}"
    if os.path.isdir(data_path):
      data_files.append(data_path)
      i += 1
    else:
      break

  print(f"found {len(data_files)} checkpoints.")
  avg_checkpoints(data_files, f"{args.model_to}.data")
  print("copying model file..")
  shutil.copyfile(args.model_from, args.model_to)
  print("Done.")

if __name__ == "__main__":
  sys.exit(main())