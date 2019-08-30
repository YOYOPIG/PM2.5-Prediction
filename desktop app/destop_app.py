import tkinter as tk  # Use tkinter as GUI
from tkinter import *
import json
import datetime as dt
from datetime import timedelta
import pytz
import requests
from display.display import Display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.interpolate as interp
import matplotlib.animation as animation 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Define helper functions
def download_data():
    for i in range(8):
        response = requests.get(f'http://140.116.82.93:6800/campus/display/{i}')
        data = json.loads(response.text)
        filename = 'pos' + str(i) + '_data.json'
        print(filename)
        with open(filename, 'w') as f:
            json.dump(data, f)
    # response = requests.get(f'http://140.116.82.93:6800/training')
    # data = json.loads(response.text)
    # with open('data.json', 'w') as f:
    #     json.dump(data, f)

def download_all_data():
    r = requests.get(f'http://140.116.82.93:6800/training')
    data = json.loads(r.text)
    with open('detail_data.json', 'w') as f:
        json.dump(data, f)

def get_pos_data(pos):
    tar_filename = 'pos' + str(pos) + '_data.json'
    with open(tar_filename) as json_file:
        data = json.load(json_file)
        for index, value in enumerate(data):
            unaware = dt.datetime.strptime(value.get('date'),  '%a, %d %b %Y %H:%M:%S %Z')
            utc_timezone = pytz.timezone('UTC')
            utc_aware = utc_timezone.localize(unaware)
            taiwan_aware = utc_aware.astimezone(pytz.timezone('Asia/Taipei'))
            value['date'] = taiwan_aware
            value['position'] = pos
        return data

def get_feat_hour_avg(pos, ft):
    # in_time = get_input_time()
    data = get_pos_data(pos)
    # df = pd.DataFrame(data)
    # # Select the duration
    # df = df.loc[ df['date'] > in_time[0] ]
    # df = df.loc[ df['date'] < in_time[1] ]
    # data = df.to_dict().values()
    # print(data)
    total_pm25_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]# size 24 for each hr(0~23)
    data_num_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    avg_pm25_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for item in data:
        total_pm25_list[item.get('date').hour] += item.get(ft)
        data_num_list[item.get('date').hour] += 1
    for i in range(24):
        avg_pm25_list[i] = total_pm25_list[i] / data_num_list[i]
    return avg_pm25_list

def get_feat_in_time(time_interval, pos, ft):
    ret_list = []
    cleaned_data = []
    tmp_ft = 0
    data_num = 0
    cur_month = time_interval[0].month
    cur_day = time_interval[0].day
    cur_hr = 0
    data = get_pos_data(pos)
    for item in data:
        if (item.get('date')>time_interval[0]) and (item.get('date')<time_interval[1]):
            cleaned_data.append(item)
    cleaned_data = cleaned_data[1:]
    if len(cleaned_data)==0:
        dt = time_interval[1] - time_interval[0]
        for i in range(int(dt.total_seconds()//3600 + 1)):
            ret_list.append(0)
        return ret_list
    # cur_month = cleaned_data[0].get('date').month
    # cur_day = cleaned_data[0].get('date').day
    # cur_hr = cleaned_data[0].get('date').hour
    for item in cleaned_data:
        if item.get('date').month!=cur_month or item.get('date').day!=cur_day or item.get('date').hour!=cur_hr:
            if data_num == 0:
                ret_list.append(0)
            else:
                ret_list.append(tmp_ft / data_num)
            tmp_ft = 0
            data_num = 0
            cur_hr += 1
            if cur_hr==24:
                cur_hr = 0
                cur_day += 1
                if cur_day==32 or (cur_day==31 and cur_month==6):
                    cur_day = 1
                    cur_month += 1
        tmp_ft += item.get(ft)
        data_num+=1
    if data_num == 0:
        ret_list.append(0)
    else:
        ret_list.append(tmp_ft / data_num)
    while time_interval[1].month>cur_month or time_interval[1].day>cur_day:
        cur_hr += 1
        if cur_hr==24:
            cur_hr = 0
            cur_day += 1
            if cur_day==32 or (cur_day==31 and cur_month==6):
                cur_day = 1
                cur_month += 1
        ret_list.append(0)
    return ret_list

def get_focus_features():
    global pm10_on
    global pm25_on
    global pm100_on
    global temp_on
    global humid_on
    feat = []
    if pm10_on.get() == True:
        feat.append('pm10')
    if pm25_on.get() == True:
        feat.append('pm25')
    if pm100_on.get() == True:
        feat.append('pm100')
    if temp_on.get() == True:
        feat.append('temp')
    if humid_on.get() == True:
        feat.append('humidity')
    return feat

def get_focus_positions():
    global pos0_on
    global pos1_on
    global pos2_on
    global pos3_on
    global pos4_on
    global pos5_on
    global pos6_on
    global pos7_on
    pos = []
    if pos0_on.get() == True:
        pos.append(0)
    if pos1_on.get() == True:
        pos.append(1)
    if pos2_on.get() == True:
        pos.append(2)
    if pos3_on.get() == True:
        pos.append(3)
    if pos4_on.get() == True:
        pos.append(4)
    if pos5_on.get() == True:
        pos.append(5)
    if pos6_on.get() == True:
        pos.append(6)
    if pos7_on.get() == True:
        pos.append(7)
    return pos

def get_input_time():
    global entry_start
    global entry_end
    taipei_tz = pytz.timezone('Asia/Taipei')
    time = []
    t1 = entry_start.get()
    t2 = entry_end.get()
    start_time = dt.datetime.strptime(t1, '%Y %m %d').replace(tzinfo=taipei_tz)
    end_time = dt.datetime.strptime(t2, '%Y %m %d').replace(tzinfo=taipei_tz)
    time.append(start_time)
    time.append(end_time)
    return time

def clear_plot():
    global graph_frame
    for widget in graph_frame.winfo_children():
        widget.destroy()

# no pos, ft one, time on
def animation_on_map():
    tar_feature =  get_focus_features()
    in_time = get_input_time()
    img = plt.imread("ncku.jpg") # background img
    ans = [] # arr of size 24, saving 24 z grids -> array to save z grids of all time
    xpos = [160, 260, 240, 60, 100, 270, 190, 350]
    ypos = [270, 255, 30, 210, 125, 120, 145, 140]
    avg_pm25_data = [] # avg_pm25_data[8pos][hr (default 24)]
    #get_feat_in_time(in_time, 0, tar_feature[0])
    for i in range(8):
        avg_pm25_data.append(get_feat_in_time(in_time, i,tar_feature[0]))
    data_num = len(avg_pm25_data[0])
    x = range(400)
    y = range(300)
    X, Y = np.meshgrid(x, y)
    z = np.zeros(120000).reshape(300, 400)
    for i in range(8):
        z[ypos[i]][xpos[i]] = 100
    z[0][:] = 1
    z[299][:] = 1
    for i in range(300):
        z[i][0] = 1
        z[i][399] = 1
    z[z==0] = np.nan
    for t in range(data_num):
        for i in range(8):
            print(i)
            z[ypos[i]][xpos[i]] = avg_pm25_data[i][t] # set the 8 positions w/ sensor
        # Interpolation
        #mask invalid values
        z = np.ma.masked_invalid(z)
        #get only the valid values
        x1 = X[~z.mask]
        y1 = Y[~z.mask]
        newarr = z[~z.mask]
        GD1 = interp.griddata((x1, y1), newarr.ravel(),
                            (X, Y),
                            method='cubic')
        ans.append(GD1)
    
    # Plot the results
    fig,ax = plt.subplots()
    # print(ans)
    # def animate2(i):
    #     ax.clear()
    #     plt.imshow(img, extent=[0, 400, 0, 300])
    #     ax.contourf(ans[i], alpha=.6)
    #     ax.set_title('Avg. PM2.5 at time %03d'%(i))
    # # canvas = FigureCanvasTkAgg(fig, graph_frame)
    # # canvas.get_tk_widget().pack()
    # interval = 0.5 #sec
    # anim = animation.FuncAnimation(fig, animate2, 24, interval=interval*1e+3,repeat_delay=1000)
    plt.imshow(img, extent=[0, 400, 0, 300])
    plt.axis('off')
    CS = ax.contourf(ans[0], alpha=.6)
    cb = fig.colorbar(CS)
    year = in_time[0].year
    month = in_time[0].month
    day = in_time[0].day
    hr = in_time[0].hour
    ax.set_title('Avg. ' + tar_feature[0] + ' at ' + str(year) + '/' + str(month) + '/' + str(day) + ' ' + str(hr))
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()
    def update_fig(t):
        ax.clear()
        plt.imshow(img, extent=[0, 400, 0, 300])
        CS = ax.contourf(ans[int(t)], alpha=.6)
        # cb.remove()
        # cb = fig.colorbar(CS)
        #time = in_time[0] + timedelta(hours=t)
        year = in_time[0].year
        month = in_time[0].month
        day = in_time[0].day
        hr = in_time[0].hour
        dh = int(t) % 24
        dd = int(int(t)/24)
        hr = in_time[0].hour + dh
        day = in_time[0].day + dd
        if hr > 23:
            hr -= 24
            day == 1
        if day==32 or (day==31 and month==6):
            day = 1
            month += 1
        ax.set_title('Avg. ' + tar_feature[0] + ' at ' + str(year) + '/' + str(month) + '/' + str(day) + ' ' + str(hr))
        canvas.draw()# = FigureCanvasTkAgg(fig, graph_frame)
    s = tk.Scale(graph_frame, label='Select time', from_=0, to=data_num-1, orient=tk.HORIZONTAL,
             length=600, showvalue=0, tickinterval=data_num-1, resolution=1, command=update_fig)
    s.pack()

# pos on, ft on, Time on (need clear data?)
def plot_line_chart():
    tar_feature =  get_focus_features()
    tar_positions = get_focus_positions()
    data = get_pos_data(tar_positions[0])
    in_time = get_input_time()
    # Add explicitly converter
    pd.plotting.register_matplotlib_converters()
    df = pd.DataFrame(data)
    # Select the duration
    df = df.loc[ df['date'] > in_time[0] ]
    df = df.loc[ df['date'] < in_time[1] ]
    # Plot y versus x(time)
    colors = ['navy', 'turquoise', 'darkorange', 'olive', 'lightgray', 'pink', 'lightgreen']
    label = tar_feature#['pm10', 'pm25', 'pm100', 'temp', 'humidity']
    label_display = tar_feature#['pm1.0', 'pm2.5', 'pm10.0', 'temperature', 'humidity']
    fig,ax = plt.subplots()
    for i in range(len(label)):
        ax.plot(df['date'], df[label[i]], c=colors[i], label=label_display[i], lw=1, ls='-')
        plt.xticks(rotation=90)
    # ax.plot(df['date'], df[label[0]], c=colors[0], label=label_display[0], lw=1, ls='-')
    # ax.plot(df['date'], df[label[1]], c=colors[1], label=label_display[1], lw=1, ls='-')
    # ax.plot(df['date'], df[label[2]], c=colors[2], label=label_display[2], lw=1, ls='-')
    # ax.plot(df['date'], df[label[3]], c=colors[3], label=label_display[3], lw=1, ls='-')
    # ax.plot(df['date'], df[label[4]], c=colors[4], label=label_display[4], lw=1, ls='-')
    #plt.xticks(x, labels, rotation='vertical')
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()
    # Uncomment this if u need to show figure in a separate window
    # plt.show()

# time on, pos on, ft on
def plot_scatter_time():
    tar_feature =  get_focus_features()
    tar_positions = get_focus_positions()
    data = get_pos_data(tar_positions[0])
    in_time = get_input_time()
    pd.plotting.register_matplotlib_converters()
    df = pd.DataFrame(data)
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
    df = df.loc[ df['date'] > in_time[0] ]
    df = df.loc[ df['date'] < in_time[1] ]
    fig,ax = plt.subplots()
    labels = ['0~6', '6~12', '12~18', '18~24']
    colors = ['navy', 'turquoise', 'darkorange', 'y']
    for i, dff in df.groupby('color'):
      ax.scatter(dff['date'], dff[tar_feature[0]], c=colors[i], label=labels[i])
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()

# time on, pos on, ft on
def plot_corr():
    tar_feature =  get_focus_features()
    tar_positions = get_focus_positions()
    in_time = get_input_time()
    data = []
    for i in tar_positions:
        data = data + get_pos_data(i)
    # convert data to dataframe
    df = pd.DataFrame(data)
    # Select the duration
    df = df.loc[ df['date'] > in_time[0] ]
    df = df.loc[ df['date'] < in_time[1] ]
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
    df = df[['month', 'day', 'weekday', 'hour_minute', column_name] + tar_feature]
    # compute the correlation
    corr = df.corr()
    # plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                vmax=0.7,
                square=True,
                annot=True,
                ax=ax,
                cmap='YlGnBu',
                linewidths=0.5)
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()

# time off, pos on, no ft
def plt_scatter():
    tar_feature =  get_focus_features()
    tar_positions = get_focus_positions()
    data = []
    for i in tar_positions:
        data = data + get_pos_data(i)
    # convert data to dataframe
    df = pd.DataFrame(data)
    # Select position 0~7
    df = df.loc[ df['position'] <= 7 ]
    # Select the duration
    #df = df.loc[ df['date'] > self.start_time ]
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
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(221)
    x = np.array(df['temp'])
    y = np.array(df['pm2.5'])
    colors = np.array(df['position'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    #ax.legend(*scatter.legend_elements(num=8), loc='upper right', title='position')
    plt.xlabel('temp (°C)')
    plt.ylabel('pm2.5 (μg/m^3)')
    # subplot 2
    ax = plt.subplot(222)
    x = np.array(df['humidity'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    #ax.legend(*scatter.legend_elements(num=8), loc='upper left', title='position')
    plt.xlabel('humidity (%)')
    plt.ylabel('pm2.5 (μg/m^3)')
    # sunplot 3
    ax = plt.subplot(223)
    x = np.array(df['hour_minute'])
    scatter = ax.scatter(x, y, c=colors, cmap='Spectral')
    #ax.legend(*scatter.legend_elements(num=8), loc='upper left', title='position')
    plt.xlabel('hour (hr.)')
    plt.ylabel('pm2.5 (μg/m^3)')
    #plt.show()
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()

# # time off, pos on, no ft
def plot_boxplot():
    tar_feature =  get_focus_features()
    tar_positions = get_focus_positions()
    data = []
    for i in tar_positions:
        data = data + get_pos_data(i)
    # convert data to dataframe
    df = pd.DataFrame(data)
    # Select position 0~7
    df = df.loc[ df['position'] <= 7 ]
    # rename the names of columns
    df = df.rename(columns = {'pm10': 'pm1.0', 'pm25': 'pm2.5', 'pm100': 'pm10.0'})
    # construct a new dataframe used to plot boxplot 1
    df_melt = pd.melt(df, id_vars=['position'], value_vars=['pm1.0', 'pm2.5', 'pm10.0'], var_name='Particulate Matter (PM)')
    # plot three boxplots
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
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
    clear_plot()
    canvas = FigureCanvasTkAgg(fig, graph_frame)
    canvas.get_tk_widget().pack()



# Main program starts here
# Create window
window = tk.Tk()
window.title('Air Quality Visualizer')
window.geometry('1000x600') # Set window size(L*W)
#window.configure(background='#42444d')
graph_frame = tk.Frame(window)

# Set fonts
bold_font = 'Open sans bold'
norm_font = 'Open sans'

# Variable declarations for ui
pos0_on = tk.BooleanVar()
pos1_on = tk.BooleanVar()
pos2_on = tk.BooleanVar()
pos3_on = tk.BooleanVar()
pos4_on = tk.BooleanVar()
pos5_on = tk.BooleanVar()
pos6_on = tk.BooleanVar()
pos7_on = tk.BooleanVar()
# features
pm10_on = tk.BooleanVar()
pm25_on = tk.BooleanVar()
pm100_on = tk.BooleanVar()
temp_on = tk.BooleanVar()
humid_on = tk.BooleanVar()
var = tk.StringVar()
plot_time = tk.IntVar()

# Declare widgets
button_dl_data = tk.Button(window, text='Download / Update data', font=(norm_font, 12), width=30, height=1, command=download_data)
button_scatter = tk.Button(window, text='Scatter', font=(norm_font, 12), width=10, height=1, command=plt_scatter)
button_line = tk.Button(window, text='Line', font=(norm_font, 12), width=10, height=1, command=plot_line_chart)
button_detail = tk.Button(window, text='Details', font=(norm_font, 12), width=10, height=1, command=plot_scatter_time)
button_corr = tk.Button(window, text='Correlation', font=(norm_font, 12), width=10, height=1, command=plot_corr)
button_box = tk.Button(window, text='Box plot', font=(norm_font, 12), width=10, height=1, command=plot_boxplot)
button_map = tk.Button(window, text='Map', font=(norm_font, 12), width=10, height=1, command=animation_on_map)

check_p0 = tk.Checkbutton(window, text='Pos0',variable=pos0_on, onvalue=1, offvalue=0)
check_p1 = tk.Checkbutton(window, text='Pos1',variable=pos1_on, onvalue=1, offvalue=0)
check_p2 = tk.Checkbutton(window, text='Pos2',variable=pos2_on, onvalue=1, offvalue=0)
check_p3 = tk.Checkbutton(window, text='Pos3',variable=pos3_on, onvalue=1, offvalue=0)
check_p4 = tk.Checkbutton(window, text='Pos4',variable=pos4_on, onvalue=1, offvalue=0)
check_p5 = tk.Checkbutton(window, text='Pos5',variable=pos5_on, onvalue=1, offvalue=0)
check_p6 = tk.Checkbutton(window, text='Pos6',variable=pos6_on, onvalue=1, offvalue=0)
check_p7 = tk.Checkbutton(window, text='Pos7',variable=pos7_on, onvalue=1, offvalue=0)

lbl_feature = tk.Label(window, text='Features',font=(bold_font, 10), width=10, height=2)
lbl_pos = tk.Label(window, text='Positions',font=(bold_font, 10), width=10, height=2)
lbl_t = tk.Label(window, text='Time interval', font=(bold_font, 10), width=12, height=2)
lbl_tf = tk.Label(window, text='(YYYY MM DD)', font=(norm_font, 10), width=12, height=2)
lbl_start = tk.Label(window, text='Start time',font=(norm_font, 10), width=10, height=2)
lbl_end = tk.Label(window, text='End time',font=(norm_font, 10), width=10, height=2)

check_pm10 = tk.Checkbutton(window, text='Pm1.0',variable=pm10_on, onvalue=1, offvalue=0)
check_pm25 = tk.Checkbutton(window, text='Pm2.5',variable=pm25_on, onvalue=1, offvalue=0)
check_pm100 = tk.Checkbutton(window, text='Pm10.0',variable=pm100_on, onvalue=1, offvalue=0)
check_temp = tk.Checkbutton(window, text='Temp',variable=temp_on, onvalue=1, offvalue=0)
check_humid = tk.Checkbutton(window, text='Humid',variable=humid_on, onvalue=1, offvalue=0)

entry_start = tk.Entry(window, show = None)
entry_end = tk.Entry(window, show = None)

# Place widgets on the window
button_dl_data.grid(row=0, column=0, columnspan=3, padx=2, pady=2)
button_scatter.grid(row=1, column=0, padx=2, pady=2)
button_line.grid(row=1, column=1, padx=2, pady=2)
button_map.grid(row=1, column=2, padx=2, pady=2)
button_detail.grid(row=2, column=0, padx=2, pady=2)
button_corr.grid(row=2, column=1, padx=2, pady=2)
button_box.grid(row=2, column=2, padx=2, pady=2)

lbl_pos.grid(row=3, column=0, padx=2, pady=2)
check_p0.grid(row=4, column=0, padx=2, pady=2)
check_p1.grid(row=4, column=1, padx=2, pady=2)
check_p2.grid(row=4, column=2, padx=2, pady=2)
check_p3.grid(row=5, column=0, padx=2, pady=2)
check_p4.grid(row=5, column=1, padx=2, pady=2)
check_p5.grid(row=5, column=2, padx=2, pady=2)
check_p6.grid(row=6, column=0, padx=2, pady=2)
check_p7.grid(row=6, column=1, padx=2, pady=2)

lbl_feature.grid(row=7, column=0, padx=2, pady=2)
check_pm10.grid(row=8, column=0, padx=2, pady=2)
check_pm25.grid(row=8, column=1, padx=2, pady=2)
check_pm100.grid(row=8, column=2, padx=2, pady=2)
check_temp.grid(row=9, column=0, padx=2, pady=2)
check_humid.grid(row=9, column=1, padx=2, pady=2)

lbl_t.grid(row=10, column=0, padx=2, pady=2)
lbl_tf.grid(row=10, column=1, padx=2, pady=2)
lbl_start.grid(row=11, column=0, padx=2, pady=2)
entry_start.grid(row=11, column=1, padx=2, pady=2)
lbl_end.grid(row=12, column=0, padx=2, pady=2)
entry_end.grid(row=12, column=1, padx=2, pady=2)

graph_frame.grid(row=0, column=3, rowspan=15, padx=5, pady=5)


# At the end, start the window loop
window.mainloop()