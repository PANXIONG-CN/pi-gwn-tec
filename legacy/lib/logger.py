import os
import logging
from datetime import datetime


def get_logger(root, name=None, debug=True, filename=None):
    """
    Create a logger writing both to console and a file. The file name can be customized.
    - root: directory to place log file
    - name: logger name (avoids duplicated handlers if reused)
    - debug: console level; file always records DEBUG
    - filename: optional file name; defaults to 'run.log'
    """
    os.makedirs(root, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # clear existing handlers to avoid duplicates when re-creating trainers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler (always write to file, with informative filename when provided)
    logfile = os.path.join(root, filename or 'run.log')
    print('Creat Log File in: ', logfile)
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log.txt', debug=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
