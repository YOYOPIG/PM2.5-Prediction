import json
import datetime as dt
import pytz
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def mean(someList):
  total = 0
  for a in someList:
    total += float(a)
  mean = total/len(someList)
  return mean
  
def standDev(someList):
  listMean = mean(someList)
  dev = 0.0
  for i in range(len(someList)):
    dev += (someList[i]-listMean)**2
  dev = dev**(1/2.0)
  return dev

def correlCo(someList1, someList2):
  # First establish the means and standard deviations for both lists.
  xMean = mean(someList1)
  yMean = mean(someList2)
  xStandDev = standDev(someList1)
  yStandDev = standDev(someList2)
  # r numerator
  rNum = 0.0
  for i in range(len(someList1)):
    rNum += (someList1[i]-xMean)*(someList2[i]-yMean)

  # r denominator
  rDen = xStandDev * yStandDev
  r =  rNum/rDen
  return r

def correlation_of_del(a,b,dict_lst):
  pos0_set = set(dict_lst[a])
  pos1_set = set(dict_lst[b])
  xlist = []
  ylist = []
  for date in pos0_set.intersection(pos1_set):
    xlist.append(dict_lst[a][date])
    ylist.append(dict_lst[b][date])
  cor = correlCo(xlist, ylist)
  return cor

def get_pos_delta(pos):
  r = requests.get(f'http://140.116.82.93:6800/campus/display/{ pos }')
  data = json.loads(r.text)
  for index, value in enumerate(data):
    unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
    utc_timezone = pytz.timezone('UTC')
    utc_aware = utc_timezone.localize(unaware)
    taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
    value['date'] = taiwan_aware
  dlist = []
  ctr = 1
  # clean data: avg per hour
  pm25_value = 0
  data_ctr = 0
  pm25_list = []
  hr_list = []
  hr = data[0].get('date').hour
  last_time = data[0].get('date').replace(minute=0, second=0, microsecond=0)
  for d in data:
    if d.get('date').hour == hr:
      pm25_value += d.get('pm25')
      data_ctr += 1
    else:
      hr = d.get('date').hour
      pm25_list.append(pm25_value/data_ctr)
      hr_list.append(last_time)
      last_time = d.get('date').replace(minute=0, second=0, microsecond=0)
      pm25_value = 0
      data_ctr = 0
      pm25_value += d.get('pm25')
      data_ctr += 1
  while ctr<len(pm25_list):
    delta = pm25_list[ctr]- pm25_list[ctr-1]
    ctr = ctr+1
    dlist.append(delta)

  return dict(zip(hr_list[1:], dlist))

dict_lst = []
for i in range(0,8):
  dict_lst.append(get_pos_delta(i))

# calculate correlation
table = []
print(correlation_of_del(0,7,dict_lst))
for i in range(0,8):
  for j in range(0,8):
    table.append(correlation_of_del(i,j,dict_lst))
A = np.array(table)
B = np.reshape(A,(-1, 8))
sns.heatmap(B,
            xticklabels=[0,1,2,3,4,5,6,7],
            yticklabels=[0,1,2,3,4,5,6,7],
            annot=True)
plt.title('Correlation of delta pm2.5 betw. different positions')
plt.show()