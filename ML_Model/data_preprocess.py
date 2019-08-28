import json
import datetime as dt
import pytz
import requests
import numpy as np

def download_training_data():
    # Download data from all 8 positions
    for i in range(8):
        response = requests.get(f'http://140.116.82.93:6800/campus/display/{i}')
        data = json.loads(response.text)
        # Change timestamp to UTC+8
        for index, value in enumerate(data):
            unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
            utc_timezone = pytz.timezone('UTC')
            utc_aware = utc_timezone.localize(unaware)
            taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
            value['month'] = taiwan_aware.month
            value['day'] = taiwan_aware.day
            value['hour'] = taiwan_aware.hour
        # Write to file
        filename = 'pos' + str(i) + '_data.json'
        cleaned_data = data_cleaning(data)
        print('Writing file : ' + filename + '... (' + str(i+1) + '/8)')
        with open(filename, 'w') as f:
            json.dump(cleaned_data, f)

def data_cleaning(raw_data):
    # Get avg data per hour, if empty try inputation
    cleaned_data = []
    cur_month = 6
    cur_day = 1
    cur_hr = 0
    ctr = 0
    tmp_pm10 = 0
    tmp_pm25 = 0
    tmp_pm100 = 0
    tmp_temp = 0
    tmp_humid = 0
    for element in raw_data:
        if(element['month']<6):
            continue
        if element['hour'] != cur_hr or element['day'] != cur_day:
            if(ctr == 0):
                avg_data = [cur_month, cur_day, cur_hr, np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                avg_data = [cur_month, cur_day, cur_hr, tmp_pm10/ctr, tmp_pm25/ctr, tmp_pm100/ctr, tmp_temp/ctr, tmp_humid/ctr]
            cleaned_data.append(avg_data)
            tmp_pm10 = 0
            tmp_pm25 = 0
            tmp_pm100 = 0
            tmp_temp = 0
            tmp_humid = 0
            ctr = 0
            cur_hr+=1
            if cur_hr==24:
                cur_hr = 0
                cur_day += 1
                if cur_day==32 or (cur_day==31 and cur_month==6):
                    cur_day = 1
                    cur_month += 1
        tmp_pm10 += element['pm10']
        tmp_pm25 += element['pm25']
        tmp_pm100 += element['pm100']
        tmp_temp += element['temp']
        tmp_humid += element['humidity']
        ctr+=1
    return cleaned_data


download_training_data()