import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Keep test execution deterministic and independent of the developer's local .env.
os.environ["PROVIDER_MODE"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-key"
