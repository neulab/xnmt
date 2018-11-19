import argparse, os, sys

from xnmt.eval import tasks
from xnmt import param_collections, utils
from xnmt.persistence import LoadSerialized, YamlPreloader, initialize_if_needed

def main() -> None:
  parser = argparse.ArgumentParser()
  utils.add_dynet_argparse(parser)
  parser.add_argument("--src", help=f"Path of source file to read from.", required=True)
  parser.add_argument("--hyp", help="Path of file to write hypothesis to.", required=True)
  parser.add_argument("--mod", help="Path of model file to read.", required=True)
  args = parser.parse_args()

  exp_dir = os.path.dirname(__file__)
  exp = "{EXP}"

  param_collections.ParamManager.init_param_col()

  # TODO: can we avoid the LoadSerialized proxy and load stuff directly?
  load_experiment = LoadSerialized(filename=args.mod)

  uninitialized_experiment = YamlPreloader.preload_obj(load_experiment, exp_dir=exp_dir, exp_name=exp)
  loaded_experiment = initialize_if_needed(uninitialized_experiment)
  model = loaded_experiment.model
  inference = model.inference
  param_collections.ParamManager.populate()

  decoding_task = tasks.DecodingEvalTask(args.src, args.hyp, model, inference)
  decoding_task.eval()

if __name__ == "__main__":
  sys.exit(main())
