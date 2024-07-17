import pyodbc
import mysql.connector
import psycopg2
import pandas as pd

from pyodbc import Error as pyodbcError
from mysql.connector import Error as mysqlError
from psycopg2 import Error as psyError

# service selector based on dropdown menu in dashboard
def switch_for_service(service, servername, database, table, username, password, port):
    if service == 'sqlserver':
        return connect_msql(servername, database, table, username, password, port)
    elif service == 'mysql':
        return connect_mysql(servername, database, table, username, password, port)
    elif service == 'postgresql':
        return connect_postgresql(servername, database, table, username, password, port)

def connect_msql(servername, database, table, username, password, port):
    try:
        conn = pyodbc.connect(Driver='{SQL Server}',
                            server=servername,
                            database=database,
                            UID=username,
                            PWD=password,
                            port=port)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table}')
        rows = cursor.fetchall()

        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=column_names)
    
        cursor.close()
        conn.close()

        return df, None
    except pyodbcError as e:
        return None, f'Connection Error with Dataserver: {e}'


def connect_mysql(servername, database, table, username, password, port):
    try:
        conn = mysql.connector.connect(host=servername,
                                    user=username,
                                    password=password,
                                    database=database,
                                    port=port
                                    )
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table}')
        rows = cursor.fetchall()

        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=column_names)

        cursor.close()
        conn.close()

        return df, None
    except mysqlError as e:
        return None, f'Connection Error with Dataserver: {e}'

def connect_postgresql(servername, database, table, username, password, port):
    try:
        conn = psycopg2.connect(host=servername,
                                port=port,
                                user=username,
                                password=password,
                                database=database
                                )
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table}')
        rows = cursor.fetchall()

        column_names = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=column_names)

        cursor.close()
        conn.close()

        return df, None
    except psyError as e: 
        return None, f'Connection Error with Dataserver: {e}'