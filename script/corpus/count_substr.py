#!/usr/bin/env python3

import sys
from collections import defaultdict

MAX = 6

substr = defaultdict(int)
for line in sys.stdin:
    char = "".join(line.strip().split(" "))
    for i in range(len(char)):
        for j in range(i+1, min(MAX+i+1, len(char))):
            substr[char[i:j]] += 1

print("Unique subst:", len(substr), file=sys.stderr)

for subs, cnt in sorted(substr.items(), key=lambda x:x[1]):
    print(subs + "\t" + str(cnt))
