#!/usr/bin/env python3
#-*- coding: UTF-8 -*-
import pandas

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class DataRead:
    def __init__(self):
        '''
            Read Question-Answer pairs

        '''
        pass

    def read(self, db_type, **args):
        '''
            Auto read data from different sources

            Input:
                - db_type: mysql or xls or csv
                - other parameters needed for different sources
        '''
        if db_type == "mysql":
            self.read_mysql(**args)
        elif db_type == "xls":
            self.read_excel(**args)
        elif db_type == "csv":
            self.read_csv(**args)


    def read_mysql(self, host, database, tablename, port, user, password, subset=None, limit = 0, offset = 0):
        '''
            Read from MySQL
            
            Input:
                - host: MySQL host
                - database: MySQL database
                - tablename: MySQL table name
                - port: MySQL port
                - user: MySQL username
                - password: MySQL password
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_
                - limit: mysql limit
                - offset: mysql offset

        '''
        import MySQLdb
        con = MySQLdb.connect(host = host,\
                port = port,\
                user = user,\
                passwd = password,\
                 charset = "utf8", use_unicode = True)
        con.select_db(database)
        
        cmd = "select * from %s"%tablename
        if limit > 0:
            cmd += " limit %i"%limit
        if offset > 0:
            cmd += " offset %i"%offset
        self.data = pandas.read_sql(cmd, con)
        self.data = self.data[header].dropna(subset=subset)
        con.close()


    def read_excel(self, filename, header, sep=',',  subset=None):
        '''
            Read from excel
            
            Input:
                - filename: file location
                - header: list of string, head names
                - sep: separator, default is ','
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        self.data = pandas.read_excel(filename, sheetname = sheetname)
        self.data.columns = header + list(self.data.columns)[len(header):]
        self.data = self.data[header].dropna(subset=subset)


    def read_csv(self, filename, header, sep=',',  subset=None):
        '''
            Read from csv
            
            Input:
                - filename: file location
                - header: list of string, head names
                - sep: separator, default is ','
                - subset: array-like, Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include. More details please check `pandas.dropna <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html>`_

        '''
        self.data = pandas.read_csv(filename, sep = sep)
        self.data.columns = header + list(self.data.columns)[len(header):]
        self.data = self.data[header].dropna(subset=subset)


