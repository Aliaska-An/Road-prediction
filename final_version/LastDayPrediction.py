import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score


# ДАТАСЕТ СОД. ДАННЫЕ ОПРЕД. СЕГМЕНТ ДОРОГИ: СКОРОСТЬ, № ДНЯ, ДЕНЬ НЕДЕЛИ, ВРЕМЯ, ЧАС
def create_LASTDAY_Dataset():
    with open('LASTDAY_Dataset.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        data1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data1.writeheader()
        with open('seg_2/LASTDAY_seg2.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                data1.writerow({'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                     'time': row['time']})

# Обработка и формирование dataset
# ДАТАСЕТ СОД. ДАННЫЕ ОПРЕД. СЕГМЕНТ ДОРОГИ: СКОРОСТЬ, № ДНЯ, ДЕНЬ НЕДЕЛИ, ВРЕМЯ, ЧАС
'''def create_initialDataset():
    with open('LASTDAY_2_initialDataset.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'time_stamp', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        data1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data1.writeheader()
        with open('LASTDAY_1_data.csv', newline='') as File:
        #with open('traffic_speed_sub-dataset', newline='') as File:
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
                data1.writerow({'road_segment_id': row[0], 'time_stamp': row[1], 'traffic_speed': row[2], 'day': int(day), 'week_day': str(week_day),'time': str(hour) +"."+ str(min), 'hour': hour, 'min': min})
                #if day == 1000:
                #    break;'''

# ДАТАСЕТ ДНЕЙ НЕДЕЛИ
def dataset_sort_weekday(weekday):
    with open('LASTDAY_3_dataset_sort_' + str(weekday) + '.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('LASTDAY_2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == weekday:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})


# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ
def dataset_sort_timeInterval():
    with open('LASTDAY_4_dataset_sort_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('LASTDAY_2_initialDataset.csv', newline='') as File:
        #with open('LASTDAY_Dataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if int(row['hour']) >= 18 and int(row['hour']) < 19:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                                  'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                                  'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (ДЕНЬ НЕДЕЛИ + ВРЕМЯ)
def dataset_sort_weekday_timeInterval():
    with open('LASTDAY_5_dataset_sort_weekday_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('LASTDAY_2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == 'wednesday' and int(row['hour']) >= 18 and int(row['hour']) < 19:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

def create_format_dataset(datasetName):
    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    dN = 'LASTDAY_2' + datasetName
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
    with open('LASTDAY_6_week_dataset.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open('LASTDAY_2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                start_day = prediction_day - 7
                day = int(row['day'])
                if day >= start_day:
                    dataset.writerow({'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'], 'time': row['time']})

# ДАТАСЕТ НЕДЕЛИ ДО ВЫБРАННОГО ДНЯ + ВРЕМЯ
def create_dataset_week_time(prediction_day):
    # ДАТАСЕТ ДЛЯ недели СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    with open('LASTDAY_6_week_dataset_time.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open('LASTDAY_2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                start_day = prediction_day - 7
                day = int(row['day'])
                if day >= start_day:
                    if int(row['hour']) >= 18 and int(row['hour']) < 19:
                        dataset.writerow({'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'], 'time': row['time']})
#-----------------------------------------------------------------------------------------------------------------------

# Обработка данных

def load_data(dataframeName):
    #print(dataframeName)
    dataframe = pd.read_csv(dataframeName)
    #print("\nDATAFRAME\n", dataframe)

    labelencoder = LabelEncoder()
    labelencoder.fit(dataframe.loc[:, 'week_day'])
    dataframe.loc[:, 'week_day'] = labelencoder.transform(dataframe.loc[:, 'week_day'])
    df_week_day = dataframe.loc[:, 'week_day']
    #print("df_week_day", df_week_day)

    labelencoder.fit(dataframe.loc[:, 'time'])
    dataframe.loc[:, 'time'] = labelencoder.transform(dataframe.loc[:, 'time'])
    #dataframe.to_csv("a.csv")

    y_data = dataframe.loc[:, 'traffic_speed']
    #y_data = y_data.apply(int)
    #for i in y_data:
        #y_data = y_data - (y_data % 5)


    dataframe.drop('traffic_speed', 1, inplace = True)
    x_data = dataframe
    #print(x_data)
    #print(y_data)
    return x_data, y_data

def metrics_errors(species, predictions):
    mse = mean_squared_error(species, predictions)
    mae = mean_absolute_error(species, predictions)
    mape = mean_absolute_percentage_error(species, predictions)
    r2 = r2_score(species, predictions)
    print(colored("\nMetrics errors", attrs=['underline']))
    print("MSE:", mse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R2:", r2)

def work(x_data, y_data):
    x_data_LASTDAY, y_data_LASTDAY = load_data("time_LASTDAY_Dataset.csv")
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.1,
                                                        random_state=42)
    model1 = RandomForestRegressor(n_estimators=500, min_weight_fraction_leaf=0,
                                   max_features='sqrt', random_state=0)
    model2 = KNeighborsRegressor(n_neighbors=4, algorithm='ball_tree', weights='distance', p=1)

    model1.fit(train_x, train_y)
    model2.fit(train_x, train_y)

    rf1_predictions = model1.predict(x_data_LASTDAY)
    rf2_predictions = model2.predict(x_data_LASTDAY)

    rf_predictions = (rf1_predictions + rf2_predictions) / 2
    species = np.array(y_data_LASTDAY)
    predictions = np.array(rf_predictions)

    ver = abs(predictions - species)
    lenght = len(ver)

    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 5:
            count = count + 1
    print("mistake: 5")

    # Вероятность
    print(count, lenght)
    print(colored(count / lenght, 'red'))
    metrics_errors(species, predictions)
    print('-------------------------------------------------------------------------------------------------------')

    return predictions

def metrics_errors(species, predictions):
    mse = mean_squared_error(species, predictions)
    mae = mean_absolute_error(species, predictions)
    mape = mean_absolute_percentage_error(species, predictions)
    r2 = r2_score(species, predictions)
    print(colored("\nMetrics errors", attrs=['underline']))
    print("MSE:", mse)
    print("MAE:", mae)
    print("MAPE:", mape)
    print("R2:", r2)

if __name__ == "__main__":
    #create_initialDataset()
    create_LASTDAY_Dataset()
    week_day = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    for iter in week_day:
        dataset_sort_weekday(iter)
    dataset_sort_timeInterval()
    dataset_sort_weekday_timeInterval()

    #приводим датасеты к формату
    # все данные
    create_format_dataset("LASTDAY_2_initialDataset.csv")
    # опред день недели
    for iter in week_day:
        name = '3_dataset_sort_' + iter + '.csv'
        create_format_dataset(name)
    # опред время
    create_format_dataset("LASTDAY_4_dataset_sort_timeInterval.csv")
    # опред день и время
    create_format_dataset("LASTDAY_5_dataset_sort_weekday_timeInterval.csv")
    create_dataset_week(60)


    # Обрабатываем данные
    x_data_LASTDAY, y_data_LASTDAY = load_data("time_LASTDAY_Dataset.csv")
    x_data2, y_data2 = load_data("LASTDAY_2LASTDAY_2_initialDataset.csv")
    x_data3, y_data3 = load_data("LASTDAY_23_dataset_sort_wednesday.csv")
    x_data6, y_data6 = load_data("LASTDAY_6_week_dataset.csv")

    print(colored("ALL DATA", 'red'))
    predictions2 = work(x_data2, y_data2)
    print(colored("WEEKDAY", 'red'))
    predictions3 = work(x_data3, y_data3)
    print(colored("WEEK", 'red'))
    predictions6 = work(x_data6, y_data6)

    print("\nPredictions2 - все время \n", predictions2)
    print("\nPredictions3 - день недели\n", predictions3)
    #print("\nPredictions6 - неделя\n", predictions6)

    #predictions_first = (predictions2 + predictions3 + predictions6)/3
    predictions_first = (predictions2 + predictions6) / 2

    ver = abs(predictions_first - y_data_LASTDAY)
    lenght = len(ver)
    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 5:
            count = count + 1

    # Вероятность
    print('\n*******************************************************************************************************\n')
    print(colored("PREDICTION FIRST:", 'red'))
    print(count, lenght)
    print(count / lenght)
    metrics_errors(y_data_LASTDAY, predictions_first)
    print('\n*******************************************************************************************************\n')


    # ПО ВРЕМЕНИ
    create_dataset_week_time(60)

    x_data4, y_data4 = load_data("LASTDAY_2LASTDAY_4_dataset_sort_timeInterval.csv")
    x_data5, y_data5 = load_data("LASTDAY_2LASTDAY_5_dataset_sort_weekday_timeInterval.csv")
    x_data7, y_data7 = load_data("LASTDAY_6_week_dataset_time.csv")


    print(colored("TIME", 'red'))
    predictions4 = work(x_data4, y_data4)
    print(colored("WEEKDAY + TIME", 'red'))
    predictions5 = work(x_data5, y_data5)
    print(colored("WEEK + TIME", 'red'))
    predictions7 = work(x_data7, y_data7)

    #predictions_second = (predictions4 + predictions5 + predictions7) / 3
    predictions_second = (predictions4 + predictions7) / 2

    ver = abs(predictions_second - y_data_LASTDAY)
    #ver = abs(predictions7 - y_data_LASTDAY)
    lenght = len(ver)
    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 5:
            count = count + 1

    # Вероятность
    print('\n*******************************************************************************************************\n')
    print(colored("PREDICTION SECOND:", 'red'))
    print(count, lenght)
    print(count / lenght)
    metrics_errors(y_data_LASTDAY, predictions_second)
    print('\n*******************************************************************************************************\n')

    #all
    #predictions_third = (predictions2 + predictions3 + predictions6 + predictions4 + predictions5 + predictions7) / 6
    predictions_third = (predictions2 + predictions6 + predictions4 + predictions7) / 4

    ver = abs(predictions_third - y_data_LASTDAY)
    lenght = len(ver)
    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 5:
            count = count + 1

    # Вероятность
    print('\n*******************************************************************************************************\n')
    print(colored("PREDICTION THIRD:", 'red'))
    print(count, lenght)
    print(count / lenght)
    metrics_errors(y_data_LASTDAY, predictions_third)
    print('\n*******************************************************************************************************\n')

