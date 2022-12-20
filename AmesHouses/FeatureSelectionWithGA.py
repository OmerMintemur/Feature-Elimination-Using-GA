import pandas as pd
import random
import numpy as np
from itertools import compress
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

population = 1000
generations = 10

average_fitness = []
best_fitness = []
best_features = []

# https://stackoverflow.com/questions/17506163/how-to-convert-a-boolean-array-to-an-int-array
# https://stackoverflow.com/questions/53527850/how-to-mask-a-list-using-boolean-values-from-another-list
df_train = pd.read_csv('TrainTransformed.csv')
print(len(df_train.columns))
df_test = pd.read_csv('TestTransformed.csv')
X_train, y_train = df_train.loc[:, df_train.columns != 'SalePrice'], df_train.loc[:, df_train.columns == 'SalePrice']
X_test, y_test = df_test.loc[:, df_test.columns != 'SalePrice'], df_test.loc[:, df_test.columns == 'SalePrice']
FULL_FEATURES = list(X_train.columns)
print(FULL_FEATURES)


class Agent:
    def __init__(self, length):
        self.features = [random.randint(0, 1) for x in range(length)]
        self.fitness = 0
        self.selected_features = []

    def __str__(self):
        self.features = np.array(self.features, dtype=bool)
        self.selected_features = list(compress(FULL_FEATURES, self.features))
        return "Fitness value of the selected features is " + str(self.fitness) + \
               "Selected features are: " + str(self.selected_features)

    def getSelectedFeatures(self):
        return list(compress(FULL_FEATURES, self.features))

    def __eq__(self, other):
        if self.features == other.features:
            return True
        else:
            return False


def models(features, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    X_train_new_features = X_train[features]
    X_test_new_features = X_test[features]

    X_train_new_features = X_train_new_features.astype('float32')
    X_test_new_features = X_test_new_features.astype('float32')

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_Normal = min_max_scaler.fit_transform(X_train_new_features)
    X_test_Normal = min_max_scaler.transform(X_test_new_features)

    reg_RF = lgb.LGBMRegressor()
    reg_RF.fit(X_train_Normal, y_train)

    # print(np.sqrt(mean_squared_error(np.log(y_test), np.log(reg_RF.predict(X_test_Normal)))))
    return np.sqrt(mean_squared_error(np.log(y_test), np.log(reg_RF.predict(X_test_Normal))))


def ga():
    agents = init_agents(population, len(FULL_FEATURES))

    for generation in range(generations):
        print('Generation ' + str(generation))

        agents = fitness(agents)
        agents = selection(agents)
        agents = my_crossover(agents)


def init_agents(population, length):
    agents = [Agent(length) for _ in range(population)]
    return agents


def fitness(agents):
    for agent in agents:
        if all(item == 0 for item in agent.features):
            ran = random.randint(0, len(agent.features) - 1)
            agent.features[ran] = 1
        selectedfeatures = agent.getSelectedFeatures()
        agent.fitness = models(selectedfeatures)
    return agents


def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    s = 0
    for agent in agents:
        s += agent.fitness
        # print("Sum is " + str(sum))

    # print(len(agents))
    average_fitness.append(s / len(agents))
    # print(average_fitness)
    best_fitness.append(agents[0].fitness)
    best_features.append(agents[0].getSelectedFeatures())
    print(agents[0].getSelectedFeatures())
    print(agents[0].fitness)
    agents = agents[:int(0.3 * len(agents))]
    return agents


def my_crossover(agents):
    offspring = []
    i = 0
    j = 1
    for c in range((population - len(agents)) // 2):
        parent1 = agents[i]
        parent2 = agents[j]

        control = 0
        while control != 2:
            child1 = Agent(len(FULL_FEATURES))
            split = random.randint(0, len(FULL_FEATURES))
            r = random.randint(0, 1)

            if r == 0:
                child1.features = parent1.features[0:split] + parent2.features[split:len(FULL_FEATURES)]
            else:
                child1.features = parent2.features[0:split] + parent1.features[split:len(FULL_FEATURES)]

            ran = random.randint(0, len(child1.features) - 1)

            if child1.features[ran] == 1:
                child1.features[ran] = 0
            else:
                child1.features[ran] = 1
            flag1 = True

            for x in offspring:
                if x.features == child1.features:
                    flag1 = False
                    break
            if flag1:
                offspring.append(child1)
                control += 1

        if j < (i+(len(agents)//20)):
            j = j + 1
        else:
            i = i + 1
            j = i + 1

    agents.extend(offspring)
    return agents


ga()
res = {'AverageFitness': average_fitness, 'BestFitness': best_fitness, 'SelectedFeatures': best_features}
df = pd.DataFrame(res)
df.to_csv("Results_Pop1000_Gen_10.csv")
print(df)
