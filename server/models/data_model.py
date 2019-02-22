from pymodm import MongoModel, fields
from db import BaseDB
class PMData(MongoModel):
    pm10 = fields.IntegerField(mongo_name="pm10")
    pm25 = fields.IntegerField(mongo_name="pm25")
    pm100 = fields.IntegerField(mongo_name="pm100")
    temp = fields.IntegerField(mongo_name="temp")
    humidity = fields.IntegerField(mongo_name="humidity")
    # position = fields.CharField()
    # date = fields.DateTimeField()
    class Meta:
        connection_alias = BaseDB.connectionAlias