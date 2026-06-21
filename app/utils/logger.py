from datetime import datetime
import logging
import os

LOG_DIR ="logs"
os.makedirs(LOG_DIR , exist_ok=True)

log_filename=os.path.join(LOG_DIR, f"sage_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger=logging.getLogger("SAGE")