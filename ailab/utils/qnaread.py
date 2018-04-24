#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
import pandas

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class QnARead:
    def __init__(self, cfg):
        '''
            Read Question-Answer pairs

            Input:
                - cfg: python dictionary or ailab.utils.config
        '''
        self.cfg = cfg


    def readQnA(self, subset=None):
        '''
            Auto read QnA from different sources

            Config keys needed:
                - question_db_type: mysql or xls or csv
                - question_header: needed table header, list of string
                - other keys needed for different sources

            Input:
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        if self.cfg["question_db_type"] == "mysql":
            self.readQnA_mysql(subset)
        elif self.cfg["question_db_type"] == "xls":
            self.readQnA_xls(subset)
        elif self.cfg["question_db_type"] == "csv":
            self.readQnA_csv(subset)


    def readQnA_mysql(self, subset=None, limit = 0, offset = 0):
        '''
            Read QnA from MySQL

            Config keys needed:
                - question_host: MySQL host
                - question_database: MySQL database
                - question_tbname: MySQL table name
                - question_port: MySQL port
                - question_user: MySQL username
                - question_password: MySQL password
            
            Input:
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        import MySQLdb
        con = MySQLdb.connect(host = self.cfg["question_host"],\
                port = self.cfg["question_port"],\
                user = self.cfg["question_user"],\
                passwd = self.cfg["question_password"],\
                 charset = "utf8", use_unicode = True)
        con.select_db(self.cfg["question_database"])
        
        cmd = "select * from %s"%self.cfg["question_tbname"]
        if limit > 0:
            cmd += " limit %i"%limit
        if offset > 0:
            cmd += " offset %i"%offset
        self.data = pandas.read_sql(cmd, con)
        self.data = self.data[self.cfg['question_header']].dropna(subset=subset)
        con.close()


    def readQnA_xls(self, subset=None):
        '''
            Read QnA from excel
            
            Config keys needed:
                - question_file: xls filename
                - question_sheetname: xls tablename
            
            Input:
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        self.data = pandas.read_excel(self.cfg["question_file"], sheetname=self.cfg['question_sheetname'])
        self.data.columns = self.cfg['question_header'] + list(self.data.columns)[len(self.cfg['question_header']):]
        self.data = self.data[self.cfg['question_header']].dropna(subset=subset)


    def readQnA_csv(self, subset=None):
        '''
            Read QnA from csv
            
            Config keys needed:
                - question_file: csv filename
                - question_sep: csv separate symbol 
            
            Input:
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        self.data = pandas.read_csv(self.cfg["question_file"], sep=self.cfg["question_sep"])
        self.data.columns = self.cfg['question_header'] + list(self.data.columns)[len(self.cfg['question_header']):]
        self.data = self.data[self.cfg['question_header']].dropna(subset=subset)


