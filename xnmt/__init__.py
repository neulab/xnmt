import os
import sys

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
  sys.path.append(package_dir)
