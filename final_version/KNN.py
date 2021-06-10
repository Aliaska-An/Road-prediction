import csv
import config
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder


def create_dataset():
    # ДАТАСЕТ СОД. ДАННЫЕ ОПРЕД. СЕГМЕНТ ДОРОГИ: СКОРОСТЬ, № ДНЯ, ДЕНЬ НЕДЕЛИ, ВРЕМЯ, ЧАС
    with open('data1.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'time_stamp', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
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
                min = int(row[1]) * 15 % 60
                hour = int(row[1]) * 15 / 60 % 24
                hour = int(hour)
                time = hour + min / 100

                data1.writerow(
                    {'road_segment_id': row[0], 'time_stamp': row[1], 'traffic_speed': row[2], 'day': int(day),
                     'week_day': str(week_day), 'time': str(hour) + "." + str(min), 'hour': hour, 'min': min})

    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (тут меняем или все дни или определенный)
    with open('dat.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'time_stamp', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('data1.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if int(row['hour']) >= 17 and int(row['hour']) <= 18:
                    # if str(row['week_day']) == 'monday':
                    dat.writerow({'road_segment_id': row['road_segment_id'], 'time_stamp': row['time_stamp'],
                                  'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                                  'time': row['time'], 'hour': row['hour'], 'min': row['min']})

    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    with open('dataset.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open('dat.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                dataset.writerow(
                    {'traffic_speed': row['traffic_speed'], 'week_day': row['week_day'], 'time': row['time']})


def load_data():
    dataframe = pd.read_csv("dataset.csv")
    print("\nDATAFRAME\n")
    #   print(dataframe)

    labelencoder = LabelEncoder()

    labelencoder.fit(dataframe.loc[:, 'week_day'])
    dataframe.loc[:, 'week_day'] = labelencoder.transform(dataframe.loc[:, 'week_day'])

    labelencoder.fit(dataframe.loc[:, 'time'])
    dataframe.loc[:, 'time'] = labelencoder.transform(dataframe.loc[:, 'time'])
    dataframe.to_csv("a.csv")

    y_data = dataframe.loc[:, 'traffic_speed']
    y_data = y_data.apply(int)
    # for i in y_data:
    # y_data = y_data - (y_data % 5)

    dataframe.drop('traffic_speed', 1, inplace=True)
    x_data = dataframe

    # print(x_data)
    # print(y_data)

    return x_data, y_data


# метод k-ближайших соседей
if __name__ == "__main__":
    create_dataset()
    x_data, y_data = load_data()
    # Создаём модель леса из сотни деревьев
    model = KNeighborsRegressor(n_neighbors=15)
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3, random_state=24)
    # Обучаем на тренировочных данных
    model.fit(train_x, train_y)
    # Действующая классификация
    rf_predictions = model.predict(test_x)
    # Вероятности для каждого класса
    rf_probs = model.predict_proba(test_x)[:, 1]

    species = np.array(test_y)
    predictions = np.array(rf_predictions)

    ver = abs(predictions - species)
    lenght = len(ver)

    count = 0
    for i in ver:
        if i <= 3:
            count = count + 1

    print(count, lenght)
    print(count / lenght)

