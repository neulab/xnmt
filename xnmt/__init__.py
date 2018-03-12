import os
import sys

# No support for python2
if sys.version_info[0] == 2:
  raise RuntimeError("XNMT does not support python2 any longer.")

if not any(a.startswith("--settings") for a in sys.argv): sys.argv.insert(1, "--settings=settings.standard")

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
  sys.path.append(package_dir)

