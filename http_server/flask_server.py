from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'pmBase'
app.config['MONGO_URI'] = 'mongodb://mongo:27017/pmBase'

mongo = PyMongo(app)

@app.route('/', methods=['GET', 'POST'])
def get_recent_data(position):
  target_data = mongo.db.pm_data
  t = target_data.find_one({'position' : position})
  if t:
    result = {'position' : t['position'], 'date' : t['date'], 'temperature' : t['temp'], 'humidity' : t['humidity'], 'pm10' : t['pm10'], 'pm25' : t['pm25'], 'pm100' : t['pm100']}
  else:
    result = "Oops! Can't find data from the sensor with that id."
  return jsonify({'search_result' : result})

if __name__ == '__main__':
    app.run(debug=True)
    