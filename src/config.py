"""
Configuration: Load environment variables.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found. "
        "Add it to your .env file."
    )