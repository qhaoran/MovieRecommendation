import sys
from pyspark import SparkContext

doc_source_path = "/Users/qhaoran/Desktop/INF-553/02_HW/HW01/ml-20m/ratings.csv"
destination_path = "/Users/qhaoran/Desktop/INF-553/02_HW/HW01/submission/Haoran_Que_result_task1_big.txt"

sc = SparkContext(appName="rating_avg")
file = sc.textFile(doc_source_path).filter(lambda line: not line[0].isalpha())
ratings_each_movie = file.map(lambda line: line.split(",")).map(lambda line: [int(line[1]), float(line[2])])
rating_sum = ratings_each_movie.aggregateByKey((0, 0), lambda U, v: (U[0] + v, U[1] + 1),
                                               lambda U1, U2: (U1[0] + U2[0], U1[1] + U2[1]))
rating_avg = rating_sum.map(lambda (x, (y, z)): (x, float(y) / z)).sortByKey(True).collect()

f = open(destination_path, 'w')
for elm in rating_avg:
    f.write(str(elm[0]) + "," + str(elm[1]) + '\n')
f.close()
