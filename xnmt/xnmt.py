# coding: utf-8
import argparse
'''
This will be the main class to perform decoding.
'''

if __name__ == "__main__":
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str)
  args = parser.parse_args()
  # Load model
  model = dy.Model()
  model_serializer = JSONSerializer()
  translator = model_serializer.load_from_file(args.model)
  # Perform decoding
  raise NotImplementedError("Decoding not implemneted yet")
