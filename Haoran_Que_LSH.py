from pyspark import SparkContext
import sys
from pyspark.mllib.linalg import Vectors

ratingFile = sys.argv[1]
outputPath = sys.argv[2]

sc = SparkContext("local")


def generateTransMatrix(iterator):
    for movie in iterator:
        dict = {}
        for user in movie[1]:
            dict[user - 1] = int(1)
        vec = Vectors.sparse(671, dict)
        yield (movie[0], vec)

def generateTransMatrixList(iterator):
    for movie in iterator:
        list = [0]*671
        for user in movie[1]:
            list[user - 1] = 1
        yield (movie[0], list)


transMatrixRaw = sc.textFile("./%s" % ratingFile).map(lambda line: line.split(",")).filter(
    lambda line: not line[0].isalpha()).map(
    lambda line: (int(line[1]), int(line[0]))).groupByKey()

transMatrix = transMatrixRaw.mapPartitions(generateTransMatrix)

b = 25
r = 5
totalHashfunc = b * r

permutations = []
for i in range(0, totalHashfunc):
    temp = []
    for index in range(0, 671):
        hashValue = (3 * index + 13 * i) % 671
        temp.append(hashValue)
    permutations.append(temp)


def minHash(iterator):
    for movieVector in iterator:
        movieId = movieVector[0]
        vector = movieVector[1]
        signatureVect = [671] * totalHashfunc
        for idx in range(0, len(vector)):
            if vector[idx] == 1.0:
                for index in range(0, totalHashfunc):
                    if permutations[index][idx] < signatureVect[index]:
                        signatureVect[index] = permutations[index][idx]
        yield (movieId, signatureVect)


signatureVector = transMatrix.mapPartitions(minHash)
signature = signatureVector.collectAsMap()

print(signature[1])
print(signature[2])


transitionMatrix = transMatrixRaw.mapPartitions(generateTransMatrixList).collectAsMap()

def getCandidatePairs(iterator):
    signatureMatrix = list(iterator)
    compared = set()
    for bandNum in range(0, b):
        for i in range(0, len(signatureMatrix)):
            for j in range(i + 1, len(signatureMatrix)):
                pair = [signatureMatrix[i][0], signatureMatrix[j][0]]
                pair = tuple(sorted(pair))
                if pair in compared:
                    break
                isIdentical = True
                s1 = signatureMatrix[i][1]
                s2 = signatureMatrix[j][1]
                for index in range(bandNum * r, bandNum * r + r):
                    if s1[index] != s2[index]:
                        isIdentical = False
                        break
                if isIdentical:
                    candidatepair = [signatureMatrix[i][0], signatureMatrix[j][0]]
                    candidatepair = tuple(sorted(candidatepair))
                    compared.add(candidatepair)
                    yield (candidatepair[0], candidatepair[1])
        print("pair-wise comparision in band %s is done" % bandNum)


def getSimilarMovieId(iterator):
    candidatePairsList = list(set(iterator))
    print("size of candidate pairs: %s" % len(candidatePairsList))
    for pair in candidatePairsList:
        # # using signature:
        # s1 = signature[pair[0]]
        # s2 = signature[pair[1]]
        # numerator = 0
        # for index in range(0, totalHashfunc):
        #     if s1[index] == s2[index]:
        #         numerator += 1
        # jaccard = float(numerator) / totalHashfunc

        # using transition matrix
        t1 = transitionMatrix[pair[0]]
        t2 = transitionMatrix[pair[1]]
        commonUserSum = 0.0
        totalRatingUserSum = 0.0
        for i in range(0, 671):
            if t1[i] == 0 and t2[i] == 0:
                continue
            if t1[i] == 1 and t2[i] == 1:
                commonUserSum += 1
                totalRatingUserSum += 1
            else:
                totalRatingUserSum += 1
        jaccard = float(commonUserSum) / totalRatingUserSum

        if (jaccard >= 0.5):
            yield ((pair[0], pair[1]), jaccard)


similarMovies = signatureVector.mapPartitions(getCandidatePairs).mapPartitions(getSimilarMovieId)
result = similarMovies.sortByKey().collect()

f = open(outputPath, "w")
for each in result:
    s = "%s,%s,%s" % (each[0][0], each[0][1], each[1])
    f.write(s)
    f.write("\n")
f.close()

predictions = sc.textFile(outputPath).map(lambda line: line.split(",")).map(lambda line: (line[0], line[1]))
groundTruth = sc.textFile("./SimilarMovies.GroundTruth.05.csv").map(lambda line: line.split(",")).map(
    lambda line: (line[0], line[1]))

intersection = predictions.intersection(groundTruth).collect()

TP = len(groundTruth.collect())
FP = len(predictions.collect()) - len(intersection)
FN = TP - len(intersection)

print("Precision: %s" % (float(TP) / (TP + FP)))
print("Recall: %s" % (float(TP) / (TP + FN)))
