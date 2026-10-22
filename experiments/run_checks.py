from __future__ import annotations

from pathlib import Path
import sys
import unittest


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(BASE / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
