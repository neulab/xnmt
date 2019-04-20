import argparse

parser = argparse.ArgumentParser()
parser.add_argument("align")
parser.add_argument("src_file")
parser.add_argument("trg_file")
args = parser.parse_args()

def read_data():
  """
  Reading alignment file, ex line:
  1-3 2-4 5-3
  """
  with open(args.align) as align_fp, \
       open(args.src_file) as src_fp, \
       open(args.trg_file) as trg_fp:
    for src_line, trg_line, align in zip(src_fp, trg_fp, align_fp):
      len_src = len(src_line.strip().split())
      len_trg = len(trg_line.strip().split())
      align = align.strip().split()
      align = [x.split("-") for x in align]
      align = [(int(f)-1, int(e)-1) for f, e in align]
      yield len_src, len_trg, align

def action_from_align(len_src, len_trg, align):
  f_to_e = {f:e for f,e in align}
  e_to_f = {e:f for f,e in align}
  # Make sure every target word is aligned
  # If not, align to src of the previous trg
  for i in range(len_trg):
    if i not in e_to_f:
      # Special case trg = 0 -> src = 0
      f_align = e_to_f[i-1] if i != 0 else 0
      align.append((f_align, i))
      e_to_f[i] = f_align
  # Make sure every source word is aligned
  # If not, align to trg of the next src
  for j in range(len_src-1, -1, -1):
    if j not in f_to_e:
      # Special case last_src -> last_trg
      e_align = f_to_e[j+1] if j != len_src-1 else len_trg-1
      align.append((j, e_align))
      f_to_e[j] = e_align
  align = sorted(align, key=lambda x: (x[1], -x[0]))
  # Selecting actions based on coverage
  f_cover = -1
  e_cover = -1
  actions = []
  for f, e in align:
    if f > f_cover:
      actions.extend(["READ"] * (f - f_cover))
      f_cover = f
    if e > e_cover:
      actions.extend(["WRITE"] * (e - e_cover))
      e_cover = e
  # Check before return
  assert len(actions) == (len_src + len_trg)
  return actions
  
def main():
  for data in read_data():
    actions = action_from_align(*data)
    print(" ".join(actions))
    
if __name__ == '__main__':
  main()
