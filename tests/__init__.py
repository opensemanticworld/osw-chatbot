from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file required for imports in tests
env_path = Path(__file__).parent.parent.parent / ".env"
env_loaded = load_dotenv(env_path, verbose=True)
if not env_loaded:
    print(f"No .env file found at {env_path}, using environment variables.")
