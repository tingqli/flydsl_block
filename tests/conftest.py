import os
import sys

# Import the package from this checkout and keep FlyDSL's disk cache out of the way:
# kernels here are closures over Python layout objects.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
