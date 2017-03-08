import sys
import math
import operator
from pyspark import SparkContext

ratingFile = sys.argv[1]
testFile = sys.argv[2]

sc = SparkContext("local")

testing_small = sc.textFile("./HW3/%s" % testFile).map(lambda line: line.split(",")).filter(
    lambda line: not line[0].isalpha()).map(
    lambda line: ((int(line[0]), int(line[1])), 1))
rating = sc.textFile("./HW3/%s" % ratingFile).map(lambda line: line.split(",")).filter(
    lambda line: not line[0].isalpha()).map(
    lambda line: ((int(line[0]), int(line[1])), float(line[2])))
join = rating.leftOuterJoin(testing_small)
rating = join.filter(lambda line: line[1][1] != 1)
ground_truth = join.filter(lambda line: line[1][1] == 1).map(
    lambda line: ((line[0][0], line[0][1]), line[1][0])).sortByKey()

# movie_bucketBy_user = [[userId, [m1, m2, ...]], ....]
movie_bucketBy_user = rating.map(lambda line: (line[0][0], line[0][1])).groupByKey().sortByKey().repartition(1)
testing_userData = testing_small.map(lambda line: (line[0][0], line[0][1])).groupByKey()

# user_ratingDict = {u1: {m1: r1, m2: r2, ...}, ...}
user_ratingDict = {}
rating_raw = rating.map(lambda line: (line[0][0], (line[0][1], line[1][0]))).groupByKey().collect()
for userBasket in rating_raw:
    ratings = {}
    for movie in userBasket[1]:
        ratings[movie[0]] = movie[1]
    user_ratingDict[userBasket[0]] = ratings

# define a map function to calculate all the weights of users
# use rdd: movie_buvketBy_user = [[userId, [m1, m2, ...]], ....]
def userWeight(iterator):
    movies4each_user = sorted(list(iterator))
    print(len(movies4each_user))
    for idx1 in range(len(movies4each_user)):
        user1_Id = movies4each_user[idx1][0]
        weightWith_user1 = {}
        for idx2 in range(idx1 + 1, len(movies4each_user)):
            user2_Id = movies4each_user[idx2][0]
            movieId_intersection = set(movies4each_user[idx1][1]).intersection(set(movies4each_user[idx2][1]))
            if len(movieId_intersection) != 0:
                corated_user1 = []
                corated_user2 = []
                for movieId in movieId_intersection:
                    corated_user1.append(user_ratingDict[user1_Id][movieId])
                    corated_user2.append(user_ratingDict[user2_Id][movieId])
                avg_user1 = sum(corated_user1) / float(len(corated_user1))
                avg_user2 = sum(corated_user2) / float(len(corated_user2))

                numerator = 0.0
                denominator1 = 0.0
                denominator2 = 0.0
                if len(corated_user1) == 1:
                    weightWith_user1[user2_Id] = 0.0  # only one co-rated movie btw u1 and u2, set their weight to 0
                else:
                    for r in range(len(corated_user1)):
                        numerator = numerator + (corated_user1[r] - avg_user1) * (corated_user2[r] - avg_user2)
                        denominator1 = denominator1 + (corated_user1[r] - avg_user1) ** 2
                        denominator2 = denominator2 + (corated_user2[r] - avg_user2) ** 2
                        if numerator == 0:
                            weightWith_user1[user2_Id] = 0.0
                        else:
                            weightWith_user1[user2_Id] = numerator / math.sqrt(denominator1 * denominator2)
            else:
                weightWith_user1[user2_Id] = 0.0  # no co-rated movies btw u1 and u2, set their weight to 0
        yield (user1_Id, weightWith_user1)


weightList = movie_bucketBy_user.mapPartitions(userWeight).collect()
weight = {}
for i in weightList:
    weight[i[0]] = i[1]
print(len(weight))

topK = 100

# calculate avg rating for active user and predicting users
def get_avg_rating(userId):
    user_rating = user_ratingDict[userId]
    avg = sum(user_rating.values()) / len(user_rating)
    return avg


def prediction(iterator):
    active_users = list(iterator)
    for user in active_users:
        active_userId = user[0]
        activeUser_movieList = user[1]

        # find top k nearest neighbors
        weight_With_activeUser = weight[active_userId]
        for Id in range(1, active_userId, 1):
            weight_With_activeUser[Id] = weight[Id][active_userId]
        # weightList_With_activeUser = [(u1,w1),(u2,w2),...]
        weightList_With_activeUser = sorted(weight_With_activeUser.items(), reverse=True, key=operator.itemgetter(1))
        topK_nearest_neighbors = weightList_With_activeUser[0: topK]

        numerator = 0.0
        denominator = 0.0
        for movieId2predict in activeUser_movieList:
            for selected_predictUser_IDwWeight in topK_nearest_neighbors:
                selected_predictUser_ID = selected_predictUser_IDwWeight[0]
                selected_predictUser_Weight = selected_predictUser_IDwWeight[1]
                Ru = get_avg_rating(selected_predictUser_ID)
                if movieId2predict in user_ratingDict[selected_predictUser_ID]:
                    numerator = numerator + (user_ratingDict[selected_predictUser_ID][
                                                 movieId2predict] - Ru) * selected_predictUser_Weight
                denominator = denominator + abs(selected_predictUser_Weight)
            Ra = get_avg_rating(active_userId)
            p = Ra + numerator / denominator
            yield ((active_userId, movieId2predict), p)


predicted_ratings = testing_userData.mapPartitions(prediction).sortByKey()

difference = predicted_ratings.join(ground_truth).map(lambda line: (line[0], abs(line[1][0] - line[1][1]))).collect()

count = [0, 0, 0, 0, 0]
square_sum = 0
for diff in difference:
    square_sum = square_sum + diff[1] ** 2
    if diff[1] < 1:
        count[0] += 1
    elif diff[1] >= 1 and diff[1] < 2:
        count[1] += 1
    elif diff[1] >= 2 and diff[1] < 3:
        count[2] += 1
    elif diff[1] >= 3 and diff[1] < 4:
        count[3] += 1
    else:
        count[4] += 1

rmse = math.sqrt(square_sum / len(difference))
print(">=0 and <1: %s" % count[0])
print(">=1 and <2: %s" % count[1])
print(">=2 and <3: %s" % count[2])
print(">=3 and <4: %s" % count[3])
print(">=4: %s" % count[4])
print("RMSE = %s" %rmse)
