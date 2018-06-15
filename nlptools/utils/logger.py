#!/usr/bin/env python
import logging

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

#LogLevel:
#0 NOTSET, 10 DEBUG, 20 INFO, 30 WARNING, 40 ERROR, 50 CRITICAL
def setLogger(appname='NLPLAB', loglevel_console=10, loglevel_file=0, logfile='', logformat='%(asctime)s | %(levelname)s: %(message)s'):
    '''
        return a logging object
        
        input:
            - appname: string, default is 'NLPLAB'
            - loglevel_console: console loglevel, default is 10
            - loglevel_file: file loglevel, default is 0
            - logfile: log file path, default is empty
            - logformat: log format, default is '%(asctime)s | %(levelname)s: %(message)s'

        output:
            - logging object

    '''
    logger = logging.getLogger(appname)
    if len(logger.handlers) > 0:
        return logger
    formatter = logging.Formatter(logformat)
    logger.setLevel(10)
    # setup console logging
    if loglevel_console > 0:
        ch = logging.StreamHandler()
        ch.setLevel(loglevel_console)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    # Setup file logging as well
    if loglevel_file > 0:
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

