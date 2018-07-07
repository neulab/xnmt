import argparse
import errno
import os
import shutil
import subprocess
import numpy

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int)
  parser.add_argument('--initial-dir', type=str)
  parser.add_argument('--tmp-basedir', type=str)
  parser.add_argument('--target-dir', type=str)
  parser.add_argument('--break-features',action='store_true', default=False)
  args = parser.parse_args()
  return args


def mkdir_p(dirname):
  try:
    os.makedirs(dirname)
  except OSError as e:
    if e.errno == errno.EEXIST and os.path.isdir(dirname):
      pass
    else:
      raise

def modify_audio(npz_original, npz_new, char_original, char_new, num_epoch, break_feats=False):
  feats = numpy.load(npz_original)
  new_feats = []
  new_size = (num_epoch % (len(feats.keys()) - 1)) + 1
  #print('epoch: %d size: %d' %(num_epoch, new_size))
  for i in range(0, new_size):
    feat = feats['arr_%d' % i]
    if break_feats:
      feat = feat[:,:20]

    new_feats.append(feat)

  numpy.savez_compressed(npz_new, *new_feats)
  subprocess.call("head -n %d %s > %s" %(new_size, char_original, char_new),shell=True)


def select_and_clear_current_version_dir(basedir):
  """
  Switch which is the current directory and remove all contents of that directory.

  Return:
    the name of the new current directory
  """
  version_dir_1 = os.path.join(basedir, 'version-1-augmentation')
  version_dir_2 = os.path.join(basedir, 'version-2-augmentation')
  current_2_flag = os.path.join(basedir, 'current_2_flag')
  current_version_dir = None
  if os.path.exists(current_2_flag):
    current_version_dir = version_dir_1
    os.remove(current_2_flag)
  else:
    current_version_dir = version_dir_2
    with open(current_2_flag, 'a') as f:
      os.utime(current_2_flag, None)

  mkdir_p(current_version_dir)
  shutil.rmtree(current_version_dir)
  mkdir_p(current_version_dir)
  return current_version_dir


def safelink(src, dst):
  src = os.path.abspath(src)
  dst = os.path.abspath(dst)
  if os.path.islink(dst):
    os.remove(dst)
  subprocess.call(["ln", "-s", src, dst])



def main():
  args = parse_args()
  tmp_basedir = args.tmp_basedir
  mkdir_p(tmp_basedir)
  current_version_dir = select_and_clear_current_version_dir(tmp_basedir)

  if args.epoch > 0:
    # do augmentation here
    data_name = 'LDC94S13A.npz'
    txt_name = 'LDC94S13A.char'
    npz_initial = os.path.abspath(os.path.join(args.initial_dir, data_name))
    txt_initial = os.path.abspath(os.path.join(args.initial_dir, txt_name))
    npz_new = os.path.abspath(os.path.join(current_version_dir, data_name))
    txt_new = os.path.abspath(os.path.join(current_version_dir, txt_name))

    modify_audio(npz_initial, npz_new, txt_initial, txt_new, args.epoch, break_feats=args.break_features)
    safelink(current_version_dir, args.target_dir)
  else:
    print('first run')
    safelink(args.initial_dir, args.target_dir)


if __name__ == "__main__":
  main()
  #print "--done--"
