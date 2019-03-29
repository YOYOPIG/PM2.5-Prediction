import json
from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo

# Create Flask app
app = Flask(__name__)

# Configuration for flask_pymongo
app.config['MONGO_DBNAME'] = 'pmBase'
app.config['MONGO_URI'] = 'mongodb://mongo:27017/pmBase'

MONGO = PyMongo(app)
if MONGO:
    print('Connected to mongo')
else:
    print('Failed connecting to mongo')

# Handle POST requests
@app.route('/', methods=['POST'])
def handle_post_rq():
    data = request.get_json()
    position = data.get('position', 'no pos')
    if position == 'no pos':
        app.logger.info('Bad request')
        return 'ERROR : Bad request, no position', 400
    else:
        return get_recent_data(position)

# Get the most recent data in the given position
# pos : The ID of the sensor to be queried
# Return value : The jsonified data that is ready to be sent, 404 if any error occured
def get_recent_data(pos):
    try:
        target_data = MONGO.db.pm_data
        # sort data backwards, gets the newest data
        return_data = list(target_data.find({'position' : pos}).sort([('_id', -1)]).limit(1))

        result = {'position' : return_data[0].get('position'), 'date' : return_data[0].get('date'),
                  'temperature' : return_data[0].get('temp'), 'humidity' : return_data[0].get('humidity'),
                  'pm10' : return_data[0].get('pm10'), 'pm25' : return_data[0].get('pm25'),
                  'pm100' : return_data[0].get('pm100')}
        return jsonify(result=result)
    except:
        app.logger.info('ERROR getting data ')
        result = "Oops! Can't find data from the sensor with that id."
        return result, 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
