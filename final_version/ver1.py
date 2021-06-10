import csv
import config
import numpy as np
import pandas as pd
from keras import models #импорт моделей
from keras import layers #импорт слоев
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true (y_true) and predicted (y_predict) values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    # Return the score
    return score


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
                min = int(row[1])*15 % 60
                hour = int(row[1])*15 / 60 % 24
                hour = int(hour)
                time = hour + min/100

                data1.writerow({'road_segment_id': row[0], 'time_stamp': row[1], 'traffic_speed': row[2], 'day': int(day), 'week_day': str(week_day),'time': str(hour) +"."+ str(min), 'hour': hour, 'min': min})

    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (тут меняем или все дни или определенный)
    with open('dat.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'time_stamp', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('data1.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if int(row['hour']) >= 17 and int(row['hour']) <= 18:
                    #if str(row['week_day']) == 'monday':
                    dat.writerow({'road_segment_id': row['road_segment_id'], 'time_stamp': row['time_stamp'], 'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'], 'time': row['time'], 'hour': row['hour'], 'min': row['min']})


    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    with open('dataset.csv', 'w') as csvfile:
        fieldnames = ['traffic_speed', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open('dat.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                dataset.writerow({'traffic_speed': row['traffic_speed'], 'week_day': row['week_day'], 'time': row['time']})


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.40, random_state=42)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


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
    #for i in y_data:
        #y_data = y_data - (y_data % 5)


    dataframe.drop('traffic_speed', 1, inplace = True)
    x_data = dataframe

    #print(x_data)
    #print(y_data)

    return x_data, y_data


if __name__ == "__main__":
    create_dataset()
    x_data, y_data = load_data()

    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3,
                                                        random_state=42)
    # Fit the training data to the model using grid search
    reg = fit_model(train_x, train_y)

    # Produce the value for 'max_depth'
    print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

    rf_predictions = reg.predict(test_x)
    #print(test_y)

    species = np.array(test_y)
    predictions = np.array(rf_predictions)

    print(species)
    print(predictions)

    ver = predictions / species
    lenght = len(ver)

    count = 0
    for i in ver:
        if i <= 1:
            count = count + 1

    print(count, lenght)
    print(count / lenght)
