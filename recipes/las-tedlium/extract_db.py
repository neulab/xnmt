"""extract_db.py: Extracts transcripts and yaml database for TEDLIUM

"""


def usage():
  print("usage: extract_db.py [options] tedlium-path data-path"
        " -h --help: print this Help message")


import sys
import getopt

import yaml
import os


class Usage(Exception):
  def __init__(self, msg):
    self.msg = msg


class ModuleTest(Exception):
  def __init__(self, msg):
    self.msg = msg


def make_parent_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
      if exc.errno != os.errno.EEXIST:
        raise


def extract(mypath, out_yaml, out_text, wavdir="{WAVDIR}/dev"):
  stm_files = [str(f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) if str(f).endswith(".stm")]
  out_data = []
  durations = []
  make_parent_dir(out_text)
  out_file_text = open(out_text, "w", encoding="utf-8")
  for stm_file in stm_files:
    f = open(mypath + "/" + stm_file, encoding="utf-8")
    for line in f:
      if not "ignore_time_segment_in_scoring" in line:
        spl = line.split()
        spk_id = spl[2]
        from_sec = float(spl[3])
        to_sec = float(spl[4])
        duration = to_sec - from_sec
        text = " ".join(spl[6:])
        assert duration > 0 and from_sec >= 0 and to_sec > 0, "bad data: " + line
        line_dict = {"wav": wavdir + "/" + stm_file[:-4] + ".wav",
                     "offset": from_sec,
                     "duration": duration,
                     "speaker_id": spk_id}
        out_data.append(line_dict)
        durations.append(duration)
        char_str = " ".join([s if s != " " else "__" for s in list(text.strip())])
        out_file_text.write(char_str + "\n")
    f.close()
  out_file_text.close()
  make_parent_dir(out_yaml)
  f_out = open(out_yaml, "w", encoding="utf-8")
  f_out.write(yaml.dump(out_data))
  f_out.close()


def main(argv=None):
  if argv is None:
    argv = sys.argv
  try:
    try:
      optlist, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError as msg:
      raise Usage(msg)
    for o, a in optlist:
      if o in ["-h", "--help"]:
        usage()
        exit(2)

    if len(args) != 2:
      raise Usage("must contain two non-optional parameters")
    tedlium_path = args[0]
    data_path = args[1]

    ###########################
    ## MAIN PROGRAM ###########
    ###########################

    extract(mypath=tedlium_path + "/dev/stm/",
            out_yaml=data_path + "/db/dev.yaml",
            out_text=data_path + "/transcript/dev.char",
            wavdir=data_path + "/wav/dev")

    extract(mypath=tedlium_path + "/test/stm/",
            out_yaml=data_path + "/db/test.yaml",
            out_text=data_path + "/transcript/test.char",
            wavdir=data_path + "/wav/test")

    extract(mypath=tedlium_path + "/train/stm/",
            out_yaml=data_path + "/db/train.yaml",
            out_text=data_path + "/transcript/train.char",
            wavdir=data_path + "/wav/train")
    ###########################
    ###########################

  except Usage as err:
    print(err.msg)
    print("for help use --help")
    return 2


if __name__ == "__main__":
  sys.exit(main())
