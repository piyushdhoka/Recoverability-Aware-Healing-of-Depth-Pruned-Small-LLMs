"""Make `import rah` work when scripts are run as `python scripts/NN_name.py` from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
