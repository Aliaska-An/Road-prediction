import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import visuals as vs
#%matplotlib inline
from sklearn.model_selection import train_test_split, ShuffleSplit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from keras.utils.np_utils import to_categorical
from sklearn.tree import DecisionTreeClassifier


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
                #print(day)
                data1.writerow({'road_segment_id': row[0], 'traffic_speed': row[2], 'day': int(day), 'week_day': str(week_day),'time': str(hour) +"."+ str(min), 'hour': hour, 'min': min})
                #if day == 1000:
                #    break;

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (тут меняем или все дни или определенный)
def dataset_sort_weekday():
    with open('3_dataset_sort_weekday.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == 'friday':
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (тут выбираем опред промежуток времени)
def dataset_sort_timeInterval():
    with open('4_dataset_sort_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if int(row['hour']) >= 17 and int(row['hour']) <= 18:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                                  'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                                  'time': row['time'], 'hour': row['hour'], 'min': row['min']})

# ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ (как дня недели так и интервала времени)
def dataset_sort_weekday_timeInterval():
    with open('5_dataset_sort_weekday_timeInterval.csv', 'w') as csvfile:
        fieldnames = ['road_segment_id', 'traffic_speed', 'day', 'week_day', 'time', 'hour', 'min']
        dat = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dat.writeheader()
        with open('2_initialDataset.csv', newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                if str(row['week_day']) == 'friday' and int(row['hour']) >= 17 and int(row['hour']) <= 18:
                    dat.writerow({'road_segment_id': row['road_segment_id'],
                              'traffic_speed': row['traffic_speed'], 'day': row['day'], 'week_day': row['week_day'],
                              'time': row['time'], 'hour': row['hour'], 'min': row['min']})

def create_format_dataset(datasetName):
    # ДАТАСЕТ ДЛЯ ОПРЕДЕЛЕННОГО ПРОМЕЖУТКА ВРЕМЕНИ СОД. СКОРОСТЬ, ДЕНЬ_НЕДЕЛИ И ВРЕМЯ
    dN = '2' + datasetName;
    print(dN)
    with open(str(dN), 'w') as csvfile:
        fieldnames = ['traffic_speed', 'day', 'week_day', 'time']
        dataset = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dataset.writeheader()
        with open(datasetName, newline='') as File:
            reader = csv.DictReader(File)
            for row in reader:
                dataset.writerow({'traffic_speed': row['traffic_speed'], 'day' : row['day'], 'week_day': row['week_day'], 'time': row['time']})

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
    print("\nDATAFRAME\n", dataframe)

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

    '''# Speed of the data
    minimum_speed = np.amin(y_data)
    print("\nMinimum speed of the data:\n", minimum_speed)
    maximum_speed = np.amax(y_data)
    print("\nMaximum speed of the data:\n", maximum_speed)
    mean_speed = np.mean(y_data)
    print("\nMean speed of the data:\n", mean_speed)
    median_speed = np.median(y_data)
    print("\nMedian speed of the data:\n", median_speed)
    std_speed = np.std(y_data)
    print("\nStandard deviation of speed of the data:\n", std_speed)'''


    dataframe.drop('traffic_speed', 1, inplace = True)
    x_data = dataframe
    #print(x_data)
    #print(y_data)
    return x_data, y_data


# Обучение модели
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# Создание графиков
def create_graphics(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(history_dict.keys())

    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true (y_true) and predicted (y_predict) values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    # Return the score
    return score


def work(x_data, y_data):
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.5,
                                                        random_state=42)
    # Подгонка обучающих данных к модели с помощью поиска по сетке
    reg = fit_model(train_x, train_y)

    # Produce the value for 'max_depth'
    #print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

    #test_x =
    rf_predictions = reg.predict(test_x)
    # print(test_y)

    species = np.array(test_y)
    predictions = np.array(rf_predictions)

    #print("\nSpeed\n", species)
    #print("\nPrediction\n", predictions)

    ver = abs(predictions - species)
    #print("\nver\n", ver)
    lenght = len(ver)

    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 3:
            count = count + 1

    # Вероятность
    print('\n###################################################################################################\n')
    print(count, lenght)
    print(count / lenght)
    print('\n###################################################################################################\n')

    return predictions



if __name__ == "__main__":
    create_initialDataset()
    dataset_sort_weekday()
    dataset_sort_timeInterval()
    dataset_sort_weekday_timeInterval()

    # все данные
    create_format_dataset("2_initialDataset.csv")
    # опред день недели
    create_format_dataset("3_dataset_sort_weekday.csv")
    # опред время
    create_format_dataset("4_dataset_sort_timeInterval.csv")
    # опред день и время
    create_format_dataset("5_dataset_sort_weekday_timeInterval.csv")

    create_dataset_week(61)

    x_data2, y_data2 = load_data("22_initialDataset.csv")
    x_data3, y_data3 = load_data("23_dataset_sort_weekday.csv")
    x_data4, y_data4 = load_data("24_dataset_sort_timeInterval.csv")
    x_data5, y_data5 = load_data("25_dataset_sort_weekday_timeInterval.csv")
    x_data6, y_data6 = load_data("6_week_dataset.csv")

    predictions2 = work(x_data2, y_data2)
    predictions3 = work(x_data3, y_data3)
    predictions4 = work(x_data4, y_data4)
    predictions5 = work(x_data5, y_data5)
    predictions6 = work(x_data6, y_data6)





''' 
    x_data, y_data = load_data()

    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.5,
                                                        random_state=42)
    # Fit the training data to the model using grid search
    reg = fit_model(train_x, train_y)

    # Produce the value for 'max_depth'
    print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

    rf_predictions = reg.predict(test_x)
    #print(test_y)

    species = np.array(test_y)
    predictions = np.array(rf_predictions)

    print("\nSpeed\n", species)
    print("\nPrediction\n", predictions)

    ver = abs(predictions - species)
    print("\nver\n", ver)
    lenght = len(ver)

    # Рассчет погрешности
    count = 0
    for i in ver:
        if i <= 5:
            count = count + 1

    # Вероятность
    print(count, lenght)
    print(count / lenght)


    create_dataset_week(61)
'''




# КЛАССИКА
''' x_data, y_data = load_data()
    
    model = build_model()
    history = train_model(x_data, y_data, model)
    create_graphics(history)'''


# Дерево решений
'''if __name__ == "__main__":
    x_data, y_data = load_data()
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.4,
                                                        random_state=25)
    dt = DecisionTreeClassifier()
    dt.fit(train_x, train_y)
    y_pred = dt.predict(test_x)


    #model = build_model()
    #print(x_data)

    print(y_pred)
    species = np.array(test_y)
    predictions = np.array(y_pred)

    ver = predictions/species
    lenght = len(ver)

    count = 0
    for i in ver:
        if i == 1:
            count = count + 1

    print(count, lenght)
    print(count/lenght)'''


#Лес случайных деревьев
'''if __name__ == "__main__":
    x_data, y_data = load_data()
    # Создаём модель леса из сотни деревьев
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.5, random_state=25)
    # Обучаем на тренировочных данных
    model.fit(train_x, train_y)
    # Действующая классификация
    rf_predictions = model.predict(test_x)
    # Вероятности для каждого класса
    rf_probs = model.predict_proba(test_x)[:, 1]

    species = np.array(test_y)
    predictions = np.array(rf_predictions)

    ver = predictions / species
    lenght = len(ver)

    count = 0
    for i in ver:
        if i == 1:
            count = count + 1

    print(count, lenght)
    print(count / lenght)

    #model = build_model()
    #print(x_data)

    #history = train_model(x_data, y_data, model)
    #create_graphics(history)'''
