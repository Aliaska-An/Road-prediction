import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from termcolor import colored, cprint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score

def create_RSDataset(road_segment):
    with open('seg_' + str(road_segment) + '.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        data1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data1.writeheader()
        with open('traffic_speed_sub-dataset', newline='') as File:
            reader = csv.reader(File)
            day = '-1'
            hour = '0'
            min = '0'
            day_week = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
            for row in reader:
                if int(row[0]) == road_segment:
                    if int(row[1]) % 96 == 0:
                        day = int(day) + 1
                        iter = day % 7
                        week_day = day_week[iter]
                        hour = '0'
                        min = '0'
                    min = int(row[1]) * 15 % 60
                    hour = int(row[1]) * 15 / 60 % 24
                    hour = int(hour)
                    time = hour + min / 100
                    print(day)
                    data1.writerow(
                        {'road_segment_id': row[0], 'traffic_speed': row[2], 'day': int(day), 'week_day': str(week_day),
                         'time': str(hour) + "." + str(min), 'hour': hour, 'min': min})
                    if int(day) == 60 and int(hour) == 23 and int(min) == 45:
                        break;

# Обработка и формирование dataset
# ДАТАСЕТ СОД. ДАННЫЕ ОПРЕД. СЕГМЕНТ ДОРОГИ: СКОРОСТЬ, № ДНЯ, ДЕНЬ НЕДЕЛИ, ВРЕМЯ, ЧАС
def create_initialDataset():
    with open('2_initialDataset.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        data1 = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data1.writeheader()
        with open('1_data.csv', newline='') as File:
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
                data1.writerow({'road_segment_id': row[0], 'traffic_speed': row[2], 'day': int(day), 'week_day': str(week_day),'time': str(hour) +"."+ str(min), 'hour': hour, 'min': min})
                if day == 1000:
                    break;

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
                if int(row['hour']) >= 18 and int(row['hour']) < 19:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                                  'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                                  'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (ДЕНЬ НЕДЕЛИ + ВРЕМЯ)
def dataset_sort_weekday_timeInterval(weekday):
    with open('5_dataset_sort_' + str(weekday) + '_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == weekday and int(row['hour']) >= 18 and int(row['hour']) < 19:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

def create_format_dataset(datasetName):
    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    dN = '2' + datasetName
    #print(dN)
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

#-----------------------------------------------------------------------------------------------------------------------

# Обработка данных
def load_data(dataframeName):
    dataframe = pd.read_csv(dataframeName)
    #print("\nDATAFRAME\n", dataframe)

    labelencoder = LabelEncoder()
    labelencoder.fit(dataframe.loc[:, 'week_day'])
    dataframe.loc[:, 'week_day'] = labelencoder.transform(dataframe.loc[:, 'week_day'])
    labelencoder.fit(dataframe.loc[:, 'time'])
    dataframe.loc[:, 'time'] = labelencoder.transform(dataframe.loc[:, 'time'])
    y_data = dataframe.loc[:, 'traffic_speed']
    #y_data = y_data.apply(int)
    #for i in y_data:
        #y_data = y_data - (y_data % 5)

    # Speed of the data
    minimum_speed = np.amin(y_data)
    maximum_speed = np.amax(y_data)
    mean_speed = np.mean(y_data)
    median_speed = np.median(y_data)
    std_speed = np.std(y_data)

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
    test_size = 0.2
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.2,
                                                        random_state=42)
    print("test size:", test_size)
    # Подгонка обучающих данных к модели с помощью поиска по сетке

    model1 = RandomForestRegressor(n_estimators=500, min_weight_fraction_leaf=0,
                                   max_features='sqrt', random_state=0)
    model1 = KNeighborsRegressor(n_neighbors=4, algorithm='ball_tree',
                                weights='distance', p=1)

    # Обучаем на тренировочных данных
    model1.fit(train_x, train_y)
    model2.fit(train_x, train_y)

    rf1_predictions = model1.predict(test_x)
    rf2_predictions = model2.predict(test_x)
    #rf3_predictions = model3.predict(test_x)
    # print(test_y)

    rf_predictions = (rf1_predictions + rf2_predictions) / 2
    species = np.array(test_y)
    predictions = np.array(rf1_predictions)

    #print("\nSpeed\n", species)
    #print("\nPrediction\n", predictions)

    ver = abs(predictions - species)
    # print("\nver\n", ver)
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


if __name__ == "__main__":
    #create_RSDataset(1144042225671)
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

    print("\nPREDICTIONS\n")
    # Обрабатываем данные
    x_data2, y_data2 = load_data("22_initialDataset.csv")
    # data 3
    predictions3 = []
    for iter in week_day:
        #dataset_sort_weekday(iter)
        name = '23_dataset_sort_' + iter + '.csv'
        x_data3, y_data3 = load_data(name)
        print(colored("WEEKDAY", 'yellow'), colored(iter, attrs=['underline']))
        predictions3 = work(x_data3, y_data3)
    #print(pd.DataFrame(predictions3))

    predictions5 = []
    for iter in week_day:
        print(colored("Day of the week: ", 'red'), colored(iter, attrs=['underline']), "\nTime: 18.00 - 19.00")
        dataset_sort_weekday(iter)
        name = '25_dataset_sort_' + iter + '_timeInterval.csv'
        x_data5, y_data5 = load_data(name)
        #print(colored("WEEKDAY + TIME", 'cyan'))
        predictions5 = work(x_data5, y_data5)
        # x_data5, y_data5 = load_data("25_dataset_sort_weekday_timeInterval.csv")'''

    x_data4, y_data4 = load_data("24_dataset_sort_timeInterval.csv")
    x_data6, y_data6 = load_data("6_week_dataset.csv")

    print(colored("ALL DATA", 'red'))
    predictions2 = work(x_data2, y_data2)
    print(colored("TIME", 'red'))
    print("18.00 - 19.00")
    predictions4 = work(x_data4, y_data4)
    #print(colored("WEEK",'red'))
    #predictions6 = work(x_data6, y_data6)