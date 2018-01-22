import sys

vocab = set()
for line in sys.stdin:
  for w in line.strip().split():
    if w not in vocab:
      vocab.add(w)

for w in vocab:
  print(w)
