import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from termcolor import colored, cprint
from matplotlib import pyplot as plt
from matplotlib import style

from numpy import genfromtxt


# Обработка и формирование dataset
# ДАТАСЕТ СОД. ДАННЫЕ ОПРЕД. СЕГМЕНТ ДОРОГИ: СКОРОСТЬ, № ДНЯ, ДЕНЬ НЕДЕЛИ, ВРЕМЯ, ЧАС
def create_initialDataset():
    with open('2_initialDataset.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        data1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data1.writeheader()

        with open('1_data.csv', newline='') as File:
            reader = csv.reader(File)
            day = '-1'
            hour = '0'
            min = '0'
            day_week = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
            for row in reader:
                if int(row[1]) % 96 == 0:
                    day = int(day) + 1
                    iter = day % 7
                    week_day = day_week[iter]
                    hour = '0'
                    min = '0'
                min = int(row[1])*15 % 60
                hour = int(row[1])*15 / 60 % 24
                hour = int(hour)
                time = hour + min/100
                #print(day)
                data1.writerow({'road_segment_id': row[0], 'traffic_speed': row[2], 'day': int(day),
                                'week_day': str(week_day),'time': str(hour) +"."+ str(min), 'hour': hour, 'min': min})

# ДАТАСЕТ ДНЕЙ НЕДЕЛИ
def dataset_sort_weekday(weekday):
    with open('3_dataset_sort_' + str(weekday) + '.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == weekday:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ
def dataset_sort_timeInterval():
    with open('4_dataset_sort_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                #if int(row['hour']) >= 18 and int(row['hour']) < 19:
                if float(row['time']) == 18.0:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                                  'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                                  'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (опред ДЕНЬ НЕДЕЛИ + ВРЕМЯ)
def dataset_sort_weekday_timeInterval(weekday):
    with open('5_dataset_sort_' + str(weekday) + '_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == weekday and int(row['hour']) >= 6 and int(row['hour']) < 7:
                #if str(row['week_day']) == weekday and float(row['time']) == 18.0:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

def create_format_dataset(datasetName):
    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    dN = '2' + datasetName
    print(dN)
    with open(str(dN), 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open(datasetName, newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                dataset.writerow({'traffic_speed': row['traffic_speed'], 'day' : row['day'], 'week_day': row['week_day'], 'time': row['time']})

# ДАТАСЕТ НЕДЕЛИ ДО ВЫБРАННОГО ДНЯ
def create_dataset_week(prediction_day):
    # ДАТАСЕТ ДЛЯ недели СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    with open('6_week_dataset.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                start_day = prediction_day - 7
                day = int(row['day'])
                if day >= start_day:
                    dataset.writerow({'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'], 'time': row['time']})


# ДАТАСЕТ ДЛЯ ГРАФИКА
def grafic_timeIntervall():
    with open('grafic_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['mean_speed', 'day']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('24_dataset_sort_timeInterval.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                    dat.writerow({'mean_speed': row['traffic_speed'], 'day': row['day']})
#-----------------------------------------------------------------------------------------------------------------------

# Обработка данных
def load_data(dataframeName):
    #print(dataframeName)
    dataframe = pd.read_csv(dataframeName)
    #print("\nDATAFRAME\n", dataframe)

    labelencoder = LabelEncoder()
    labelencoder.fit(dataframe.loc[:, 'week_day'])
    dataframe.loc[:, 'week_day'] = labelencoder.transform(dataframe.loc[:, 'week_day'])
    labelencoder.fit(dataframe.loc[:, 'time'])
    dataframe.loc[:, 'time'] = labelencoder.transform(dataframe.loc[:, 'time'])
    #print("\nDATAFRAME\n", dataframe)

    y_data = dataframe.loc[:, 'traffic_speed']

    # Speed of the data
    minimum_speed = np.amin(y_data)
    print("\nMinimum speed of traffic:", colored(minimum_speed, 'red'))
    maximum_speed = np.amax(y_data)
    print("Maximum speed of traffic:", colored(maximum_speed, 'red'))
    mean_speed = np.mean(y_data)
    print("Mean speed of traffic:", colored(mean_speed, 'red'))
    median_speed = np.median(y_data)
    print("Median speed of traffic:", colored(median_speed, 'red'))
    std_speed = np.std(y_data)
    print("Standard deviation of speed of traffic:", colored(std_speed, 'red'))
    print("\n\n")

    dataframe.drop('traffic_speed', 1, inplace = True)
    x_data = dataframe
    #print(x_data)
    #print(y_data)
    #x_data.to_csv("x_data.csv")
    #y_data.to_csv("x_data.csv")
    return x_data, y_data

def create_diagram(name):
    '''plt.title('Epic Info')
    plt.ylabel('speed')
    plt.xlabel('days')
    data = np.genfromtxt(name, delimiter=',', names=['day', 'mean_speed'])
    plt.plot(data['mean_speed'], data['day'])
    plt.show()'''
    data = np.loadtxt(name)
    plt.hist(data, normed=True, bins='auto')
    return 0

if __name__ == "__main__":
    create_initialDataset()
    week_day = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    for iter in week_day:
        dataset_sort_weekday(iter)
    dataset_sort_timeInterval()
    for iter in week_day:
        dataset_sort_weekday_timeInterval(iter)

    #приводим датасеты к формату
    # все данные
    create_format_dataset("2_initialDataset.csv")
    # опред день недели
    for iter in week_day:
        name = '3_dataset_sort_' + iter + '.csv'
        create_format_dataset(name)
    # опред время
    create_format_dataset("4_dataset_sort_timeInterval.csv")

    # опред день и время
    for iter in week_day:
        name = '5_dataset_sort_' + iter + '_timeInterval.csv'
        create_format_dataset(name)
    create_dataset_week(61)


    # Обрабатываем данные
    # Все данные
    '''x_data2, y_data2 = load_data("22_initialDataset.csv")
    # data 3
    print("\nDATA 3 -------------------------------------------------------\n")
    predictions3 = []
    for iter in week_day:
        print("\nDay of the week: ",  colored(iter, attrs=['underline']))
        dataset_sort_weekday(iter)
        name = '23_dataset_sort_' + iter + '.csv'
        x_data3, y_data3 = load_data(name)'''

    # Даные времени
    print("\nDay of the week:", colored("all days", attrs=['underline']),"\nTime: 4.00 - 5.00")
    x_data4, y_data4 = load_data("24_dataset_sort_timeInterval.csv")
    print("\nDATA 5 -------------------------------------------------------\n")
    predictions5 = []
    for iter in week_day:
        print("\nDay of the week: ", colored(iter, attrs=['underline']), "\nTime: 4.00 - 5.00")
        dataset_sort_weekday(iter)
        name = '25_dataset_sort_' + iter + '_timeInterval.csv'
        x_data5, y_data5 = load_data(name)
    #x_data6, y_data6 = load_data("6_week_dataset.csv")
    #grafic_timeIntervall()
    #create_diagram('ggg.csv')
