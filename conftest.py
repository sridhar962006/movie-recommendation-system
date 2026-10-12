"""Makes the project root importable when running `pytest` from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
