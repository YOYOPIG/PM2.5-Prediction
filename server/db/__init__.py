from pymodm.connection import connect
from .baseDB import BaseDB

def connectToDB():
    connect(BaseDB.base_url, BaseDB.connectionAlias)