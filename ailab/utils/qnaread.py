#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
import pandas

class QnARead:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def readQnA(self, subset=None):
        if self.cfg["question_db_type"] == "mysql":
            self.readQnA_mysql(subset)
        elif self.cfg["question_db_type"] == "xls":
            self.readQnA_xls(subset)
        elif self.cfg["question_db_type"] == "csv":
            self.readQnA_csv(subset)

    def readQnA_mysql(self, subset=None, limit = 0, offset = 0):
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
        self.data = pandas.read_excel(self.cfg["question_file"], sheetname=self.cfg['question_sheetname'])
        self.data.columns = self.cfg['question_header'] + list(self.data.columns)[len(self.cfg['question_header']):]
        self.data = self.data[self.cfg['question_header']].dropna(subset=subset)
    
    def readQnA_csv(self, subset=None):
        self.data = pandas.read_csv(self.cfg["question_file"], sep=self.cfg["question_sep"])
        self.data.columns = self.cfg['question_header'] + list(self.data.columns)[len(self.cfg['question_header']):]
        self.data = self.data[self.cfg['question_header']].dropna(subset=subset)


