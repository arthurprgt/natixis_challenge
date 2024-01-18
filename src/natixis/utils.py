"""Create utils for the project"""

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add logger to your code
logger = logging.getLogger(__name__)
