import os
import logging
from datetime import datetime

def setup_logger(output_dir, exp_name):
    """"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{exp_name}_{timestamp}.log")
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_path