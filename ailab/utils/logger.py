#!/usr/bin/env python
import logging

def setLogger(cfg_input):
    cfg = {'APPNAME': 'NLPLAB',\
            'LogLevel_Console': 10, \
            'LogLevel_File':0, \
            'LogFile':'',\
            'LogFormat': '%(asctime)s | %(levelname)s: %(message)s'}
    for k in cfg_input: cfg[k] = cfg_input[k]
    logger = logging.getLogger(cfg["APPNAME"])
    formatter = logging.Formatter(cfg["LogFormat"])
    logger.setLevel(10)
    # setup console logging
    if cfg["LogLevel_Console"] > 0:
        ch = logging.StreamHandler()
        ch.setLevel(cfg["LogLevel_Console"])
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    # Setup file logging as well
    if cfg["LogLevel_File"] > 0:
        fh = logging.FileHandler(cfg["LogFile"])
        fh.setLevel(cfg["LogLevel_File"])
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

