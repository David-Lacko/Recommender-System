import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error

import operator
import random


class RecommenderSystem(object):

    def __init__(self, data, metric, algorithm, user):
        self.data = data

        self.distance, self.neighbors = self.__find_neighbors(metric, algorithm)
        self.recommended_tuple = self.__recommend(user)

    @staticmethod
    def __get_neighbors(indices, distance, df):
        all_neighbors = []
        for x in range(len(indices)):
            neighbors = []
            for i in range(len(indices[x])):
                neighbors.append((df.iloc[indices[x, i], 0], distance[x, i]))
            all_neighbors.append(neighbors)
        return all_neighbors

    def __find_neighbors(self, metric, algorithm, n_neighbors=5):
        knn = NearestNeighbors(metric=metric, p=2, algorithm=algorithm)
        knn.fit(self.data.iloc[:, :5].values)
        distance, indices = knn.kneighbors(self.data.iloc[:, :5].values, n_neighbors=n_neighbors)

        return distance, self.__get_neighbors(indices, distance, self.data.iloc[:, :5])

    def __recommend(self, user):
        user_games = self.data[self.data['user-id'] == user]
        dissim_games = []

        for neighbor in self.neighbors[self.data.index[self.data["user-id"] == user].tolist()[0]]:
            temp = self.data[(self.data['user-id'] == neighbor[0]) & (~self.data['game-title'].isin(user_games['game-title']))]

            for index, game in temp.iterrows():
                dissim_games.append((game['game-title'], game['rating']))
        dissim_games.sort(key=operator.itemgetter(0))

        flag = ""
        rec_list, running_sum, count = [], 0, 0

        for dis in dissim_games:
            if flag != dis[0]:
                if flag != "":
                    rec_list.append((flag, running_sum / count))
                flag = dis[0]
                running_sum = dis[1]
                count = 1

            else:
                running_sum += dis[1]
                count += 1

        sort_list = sorted(rec_list, key=operator.itemgetter(1), reverse=True)
        return (sort_list)

    def __rec_games(self):
        games = []
        for pair in self.recommended_tuple:
            if pair[1] > 3.8:
                games.append(pair[0])
        return games

    def execute_system(self):
        recommendations = self.__rec_games()
        return recommendations

    def evaluate(self, user):
        errors_lst = []
        user_r = self.data[self.data["user-id"] == user]
        for i in self.neighbors[self.data.index[self.data["user-id"] == user].tolist()[0]]:
            neighbor = self.data[self.data["user-id"] == i[0]]
            ma = mean_absolute_error(pd.merge(user_r, neighbor, how="inner", on="game-title")["rating_x"],
                                     pd.merge(user_r, neighbor, how="inner", on="game-title")["rating_y"])
            errors_lst.append(ma)

        mean_error = np.mean(errors_lst)
        accuracy = (100 / 5) * (5 - mean_error)

        return mean_error, accuracy


if __name__ == "__main__":
    data = pd.read_csv("Data/steam_processed.csv", usecols=["user-id", "game-id", "hours-played", "frequency",
                                                            "rating", "game-title"])
    np.random.seed(12345)
    users = random.sample(set(data["user-id"]), k=3)
    trained_data = data[["user-id", "game-id", "hours-played", "frequency", "rating", "game-title"]]
    recommendations = []
    users_acc = []
    for user in users:
        RS = RecommenderSystem(data=trained_data, metric="minkowski", algorithm="auto", user=user)
        recommendation = RS.execute_system()
        recommendations.append(recommendation)

        users_acc.append(RS.evaluate(user)[1])
        print("\n------------------------------------\n")
        print(f"Mean Absolute Error: {RS.evaluate(user)[0]}")
        print(f"System Accuracy: {RS.evaluate(user)[1]}")
        print(f"User: {user}")


    print("\n------------------------------------\n")
    print(recommendations)
    print("\n------------------------------------\n")
    print(f"Mean Accuracy:{np.mean(users_acc)}")




















