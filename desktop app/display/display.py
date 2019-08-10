import json
import datetime as dt
import pytz
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Display():
  def __init__(self, pos, time):
    self.pos = pos
    taipei_tz = pytz.timezone('Asia/Taipei')
    self.start_time = dt.datetime.strptime(time[0], '%Y %m %d').replace(tzinfo=taipei_tz)
    self.end_time = dt.datetime.strptime(time[1], '%Y %m %d').replace(tzinfo=taipei_tz)
    self.index = 0
  def get_data(self):
    print(self.pos[self.index])
    r = requests.get(f'http://140.116.82.93:6800/campus/display/{ self.pos[self.index] }')
    # date field in self.data is the str of datetime
    # We need to convert it to timezone aware object first
    self.data = json.loads(r.text)
    for index, value in enumerate(self.data):
      # strptime() parse str of date according to the format given behind
      # It is still naive datetime object, meaning that it is unaware of timezone
      unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
      # Create a utc timezone
      utc_timezone = pytz.timezone('UTC')
      # make utc_unaware obj aware of timezone
      # Convert the given time directly to literally the same time with different timezone
      # For example: Change from 2019-05-19 07:41:13(unaware) to 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC)
      utc_aware = utc_timezone.localize(unaware)
      # This can also do the same thing
      # Replace the tzinfo of an unaware datetime object to a given tzinfo
      # utc_aware = unaware.replace(tzinfo=pytz.utc)

      # Transform utc timezone to +8 GMT timezone
      # Convert the given time to the same moment of time just like performing timezone calculation
      # For example: Change from 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC) to 2019-05-19 15:41:13+08:00(aware, tzinfo=Asiz/Taipei)
      taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
      # print(f"{ index }: {unaware} {utc_aware} {taiwan_aware}")
      value['date'] = taiwan_aware
  def get_all_data(self):
    r = requests.get(f'http://140.116.82.93:6800/training')
    # date field in self.data is the str of datetime
    # We need to convert it to timezone aware object first
    self.data = json.loads(r.text)
    for index, value in enumerate(self.data):
      # strptime() parse str of date according to the format given behind
      # It is still naive datetime object, meaning that it is unaware of timezone
      unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
      # Create a utc timezone
      utc_timezone = pytz.timezone('UTC')
      # make utc_unaware obj aware of timezone
      # Convert the given time directly to literally the same time with different timezone
      # For example: Change from 2019-05-19 07:41:13(unaware) to 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC)
      utc_aware = utc_timezone.localize(unaware)
      # This can also do the same thing
      # Replace the tzinfo of an unaware datetime object to a given tzinfo
      # utc_aware = unaware.replace(tzinfo=pytz.utc)

      # Transform utc timezone to +8 GMT timezone
      # Convert the given time to the same moment of time just like performing timezone calculation
      # For example: Change from 2019-05-19 07:41:13+00:00(aware, tzinfo=UTC) to 2019-05-19 15:41:13+08:00(aware, tzinfo=Asiz/Taipei)
      taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
      # print(f"{ index }: {unaware} {utc_aware} {taiwan_aware}")
      value['date'] = taiwan_aware

  def get_dl_data(self,pos):
    tar_filename = 'pos' + str(pos) + '_data.json'
    with open(tar_filename) as json_file:
      data = json.load(json_file)
      for index, value in enumerate(data):
        unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
        utc_timezone = pytz.timezone('UTC')
        utc_aware = utc_timezone.localize(unaware)
        taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
        value['date'] = taiwan_aware

  def plt_scatter_time(self):
    # Add explicitly converter
    pd.plotting.register_matplotlib_converters()
    df = pd.DataFrame(self.data)
    color_arr = []
    for item in df['date']:
      if item.hour >= 6 and item.hour < 12:
        color_arr.append(1)
      elif item.hour >= 12 and item.hour < 18:
        color_arr.append(2)
      elif item.hour >= 18 and item.hour < 24:
        color_arr.append(3)
      else: # 00 ~ 06 early in the morning
        color_arr.append(0)
    # Set color_arr to the third column of df for colouring
    df['color'] = color_arr
    # Select the duration
    df = df.loc[ df['date'] > self.start_time ]
    df = df.loc[ df['date'] < self.end_time ]
    plt.figure(figsize=(20, 20))
    labels = ['0~6', '6~12', '12~18', '18~24']
    colors = ['navy', 'turquoise', 'darkorange', 'y']
    for i, dff in df.groupby('color'):
      plt.scatter(dff['date'], dff['pm25'], c=colors[i], label=labels[i])
  def create_graph(self):
    plt.title('pm2.5 plot')
    plt.xlabel('Date', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel('pm2.5 (μg/m^3)')
    plt.legend()
    plt.show()
  def reset(self):
    self.pos = -1
    self.data = []
    plt.close()

  def plt_figure(self):
    plt.figure(figsize=(20, 20))

  def plt_multiple_pos(self):
    # Add explicitly converter
    pd.plotting.register_matplotlib_converters()
    df = pd.DataFrame(self.data)
    # Select the duration
    df = df.loc[ df['date'] > self.start_time ]
    df = df.loc[ df['date'] < self.end_time ]
    # Plot y versus x(time)
    colors = ['navy', 'turquoise', 'darkorange', 'olive', 'lightgray', 'pink', 'lightgreen', 'black']
    label = 'position %d' % self.pos[self.index]
    plt.plot(df['date'], df['pm25'], c=colors[self.index], label=label, lw=1, ls='-') # marker = '.' , alpha=0.8
    self.index = self.index + 1
  
  def plt_multiple_features(self):
    # Add explicitly converter
    pd.plotting.register_matplotlib_converters()
    df = pd.DataFrame(self.data)
    # Select the duration
    df = df.loc[ df['date'] > self.start_time ]
    df = df.loc[ df['date'] < self.end_time ]
    # Plot y versus x(time)
    colors = ['navy', 'turquoise', 'darkorange', 'olive', 'lightgray', 'pink', 'lightgreen']
    label = ['pm10', 'pm25', 'pm100', 'temp', 'humidity']
    label_display = ['pm1.0', 'pm2.5', 'pm10.0', 'temperature', 'humidity']
    plt.plot(df['date'], df[label[0]], c=colors[0], label=label_display[0], lw=1, ls='-')
    plt.plot(df['date'], df[label[1]], c=colors[1], label=label_display[1], lw=1, ls='-')
    plt.plot(df['date'], df[label[2]], c=colors[2], label=label_display[2], lw=1, ls='-')
    plt.plot(df['date'], df[label[3]], c=colors[3], label=label_display[3], lw=1, ls='-')
    plt.plot(df['date'], df[label[4]], c=colors[4], label=label_display[4], lw=1, ls='-')
    self.index = self.index + 1

  def combine_df(self):
    df = pd.DataFrame(self.data)
    df = df.tail(15)
    # add position column in the dataframe
    df['pos'] = self.pos[self.index]
    if self.index == 0:
      self.df = df
    else:
      self.df = pd.concat([self.df, df])
    # increment the index value
    self.index = self.index + 1
    
  def print_recent_data(self):
    # convert data to dataframe
    self.df = pd.DataFrame(self.data)
    # set the order of the columns
    self.df = self.df[['date', 'pm10', 'pm25', 'pm100', 'temp', 'humidity', 'position']]
    # set that display at most 300 rows in the dataframe
    pd.set_option('display.max_rows', 300)
    print(self.df.tail(300))

  def plt_corr(self):
    # convert data to dataframe
    df = pd.DataFrame(self.data)
    # Add columns for month, day, weekday, hour_minute
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday)
    df['hour_minute'] = df['date'].apply(lambda x: x.hour+x.minute/60)
    # Add a column that equals to hour_minute-shift_value
    shift_value = 11
    plus_value = 24 + shift_value
    column_name = 'hour_minute_minus%d' % shift_value
    df[column_name] = df['hour_minute'].apply(lambda x: x-shift_value)
    df[column_name] = df[column_name].apply(lambda x: x+plus_value if x<0 else x)
    # set the order of the columns
    df = df[['month', 'day', 'weekday', 'hour_minute', column_name, 'pm10', 'pm25', 'pm100', 'temp', 'humidity', 'position']]
    # compute the correlation
    corr = df.corr()
    # plot correlation matrix
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                vmax=0.7,
                square=True,
                annot=True,
                ax=ax,
                cmap='YlGnBu',
                linewidths=0.5)
    plt.show()

  def plt_boxplot(self):
    # convert data to dataframe
    df = pd.DataFrame(self.data)
    # Select position 0~7
    df = df.loc[ df['position'] <= 7 ]
    # rename the names of columns
    df = df.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
    # construct a new dataframe used to plot boxplot 1
    df_melt = pd.melt(df, id_vars=['position'], value_vars=['pm1.0', 'pm2.5', 'pm10.0'], var_name='Particulate Matter (PM)')
    # plot three boxplots
    fig1, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 8))
    # subplot 1
    ax = sns.boxplot(x='position', y='value', data=df_melt, hue='Particulate Matter (PM)', palette='Set3', ax=axes[0])
    ax.axis(ymin=0, ymax=100)
    ax.set_xlabel('')
    ax.set_ylabel('(μg/m^3)')
    # subplot 2
    ax = sns.boxplot(x='position', y='temp', data=df, color='orange', ax=axes[1])
    ax.axis(ymin=20, ymax=40)
    ax.set_xlabel('')
    ax.set_ylabel('temp(°C)')
    # subplot 3
    ax = sns.boxplot(x='position', y='humidity', data=df, color='cyan', ax=axes[2])
    ax.axis(ymin=15, ymax=100)
    ax.set_ylabel('humidity(%)')
    plt.show()

  def plt_scatter(self):
    # convert data to dataframe
    df = pd.DataFrame(self.data)
    # Select position 0~7
    df = df.loc[ df['position'] <= 7 ]
    # Select the duration
    df = df.loc[ df['date'] > self.start_time ]
    # rename the names of columns
    df = df.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
    # Add a column for hour_minute
    df['hour_minute'] = df['date'].apply(lambda x: x.hour+x.minute/60)
    # set the order of the columns & discard some columns
    df = df[['hour_minute', 'pm1.0', 'pm2.5', 'pm10.0', 'temp', 'humidity', 'position']]
    # exclude outliers
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    # plot scatter plot
    # subplot 1
    ax = plt.subplot(221)
    x = np.array(df['temp'])
    y = np.array(df['pm2.5'])
    colors = np.array(df['position'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    ax.legend(*scatter.legend_elements(num=8), loc='upper right', title='position')
    plt.xlabel('temp (°C)')
    plt.ylabel('pm2.5 (μg/m^3)')
    # subplot 2
    ax = plt.subplot(222)
    x = np.array(df['humidity'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    ax.legend(*scatter.legend_elements(num=8), loc='upper left', title='position')
    plt.xlabel('humidity (%)')
    plt.ylabel('pm2.5 (μg/m^3)')
    # sunplot 3
    ax = plt.subplot(223)
    x = np.array(df['hour_minute'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    ax.legend(*scatter.legend_elements(num=8), loc='upper left', title='position')
    plt.xlabel('hour (hr.)')
    plt.ylabel('pm2.5 (μg/m^3)')
    plt.show()

  def plt_test(self):
    # convert data to dataframe
    
    df = pd.DataFrame(self.data)
    print(df['pm25'])
    # # Add columns for month, day, weekday, hour_minute
    # df['month'] = df['date'].apply(lambda x: x.month)
    # df['day'] = df['date'].apply(lambda x: x.day)
    # df['weekday'] = df['date'].apply(lambda x: x.weekday)
    # df['hour_minute'] = df['date'].apply(lambda x: x.hour+x.minute/60)
    # # Add a column that equals to hour_minute-shift_value
    # shift_value = 11
    # plus_value = 24 + shift_value
    # column_name = 'hour_minute_minus%d' % shift_value
    # df[column_name] = df['hour_minute'].apply(lambda x: x-shift_value)
    # df[column_name] = df[column_name].apply(lambda x: x+plus_value if x<0 else x)
    # # set the order of the columns
    # df = df[['month', 'day', 'weekday', 'hour_minute', column_name, 'pm10', 'pm25', 'pm100', 'temp', 'humidity', 'position']]
    # # compute the correlation
    # corr = df.corr()
    # # plot correlation matrix
    # fig, ax = plt.subplots(figsize=(7, 7))
    # sns.heatmap(corr, 
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values,
    #             vmax=0.7,
    #             square=True,
    #             annot=True,
    #             ax=ax,
    #             cmap='YlGnBu',
    #             linewidths=0.5)
    # plt.show()
