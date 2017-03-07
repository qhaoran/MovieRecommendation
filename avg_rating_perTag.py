import sys
import os
from pyspark import SparkContext, sql
from pyspark.sql import SQLContext, Row
import csv

tag_path = '/Users/qhaoran/Desktop/INF-553/02_HW/HW01/ml-20m/tags.csv'
rating_path = '/Users/qhaoran/Desktop/INF-553/02_HW/HW01/ml-20m/ratings.csv'
saveRDD2path = "result1"
destination_path = "/Users/qhaoran/Desktop/INF-553/02_HW/HW01/submission/Haoran_Que_task2_big.csv"

result_path = "./%s/p*" % saveRDD2path
reload(sys)
sys.setdefaultencoding('utf-8')
sc = SparkContext(appName="rating_per_tag")
sqlContext = SQLContext(sc)
tag_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
    tag_path).drop("userId").drop("timestamp")
rating_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
    rating_path).drop("userId").drop("timestamp")

joined = tag_df.join(rating_df, "movieId", "outer").drop("movieId")
tag_avg = joined.groupby("tag").agg({"rating": "mean"})
tag_avg = tag_avg.orderBy(tag_avg.tag.desc()).collect()

tag_avg_csv = sqlContext.createDataFrame(tag_avg, ["tag", "rating_avg"])

tag_avg_csv.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(saveRDD2path)
os.system("cat %s >> %s" % (result_path, destination_path))
