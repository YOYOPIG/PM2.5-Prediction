from pymodm.connection import connect
from .base_db import BaseDB

def connect_to_db():
    connect(BaseDB.base_url, BaseDB.connectionAlias)
