import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

import scala.math._
import java.io._

import org.apache.spark.rdd.RDD

/**
  * Created by qhaoran on 2017/3/3.
  */

object task1 {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("task1").setMaster("local")
    val sc = new SparkContext(conf)

    val testingFile_small = sc.textFile("/Users/qhaoran/IdeaProjects/553HW3/testing_small.csv")
    val ratingFile_small = sc.textFile("/Users/qhaoran/IdeaProjects/553HW3/ratings_small.csv")
    val testingFile_big = sc.textFile("/Users/qhaoran/IdeaProjects/553HW3/testing_20m.csv")
    val ratingFile_big = sc.textFile("/Users/qhaoran/IdeaProjects/553HW3/ratings_big.csv")

//    val (predictionSmall, differenceSmall, rmseSmall) = rating_pred(testingFile_small, ratingFile_small, 5, 10)
//    write_result("Haoran_Que_result_task1_small.txt", predictionSmall, differenceSmall, rmseSmall)


    val (predictionBig, differenceBig, rmseBig) = rating_pred(testingFile_big, ratingFile_big, 5, 10)
    write_result("Haoran_Que_result_task1_big.txt", predictionBig, differenceBig, rmseBig)

  }

  def rating_pred(testFile: RDD[String], ratingFile: RDD[String], rank: Int, numIterations: Int): (RDD[(Int, Int, Double)], Array[Double], Double) = {
    val testingRDD = testFile.map(line => line.split(",")).filter(line => !(line(0) == "userId")).map(line => ((line(0).toInt, line(1).toInt), 1))
    val join = ratingFile.map(line => line.split(",")).filter(line => !(line(0) == "userId")).map(line => ((line(0).toInt, line(1).toInt), line(2).toDouble)).leftOuterJoin(testingRDD)
    val ratingRDD = join.filter(line => line._2._2.isInstanceOf[None.type]).map(line => (line._1._1, line._1._2, line._2._1) match {
      case (user: Int, movie: Int, rating: Double) => Rating(user, movie, rating)
    })
    val groundTruth = join.filter(line => line._2._2.isInstanceOf[Some[Int]]).map(line => (line._1._1, line._1._2, line._2._1) match {
      case (user: Int, movie: Int, rating: Double) => Rating(user, movie, rating)
    })
    val userMovie = groundTruth.map { case Rating(user, movie, rating) => (user, movie) } // 20256


    //    val rank = 5
    //    val numIterations = 10
    val model = ALS.train(ratingRDD, rank, numIterations, 0.01)
    var predictions = model.predict(userMovie).map { case Rating(user, movie, rating) => ((user, movie), rating) } //18733


    //    missing value
    val totalTest = groundTruth.map {
      case Rating(user, movie, rating) => ((user, movie), rating)
    }
    val sumRating = ratingRDD.map { case Rating(user, movie, rating) => (1, rating) }.reduceByKey(_ + _).collect()
    val avgRating = sumRating(0)._2 / ratingRDD.count()
    val missingPart = groundTruth.map { case Rating(user, movie, rating) => ((user, movie), 1.0) }.subtract(predictions.map(line => (line._1, 1.0)))
      .map(line => (line._1, avgRating))
    predictions = predictions.union(missingPart).sortByKey()
    val join_w_trueRating = groundTruth.map { case Rating(user, movie, rating) => ((user, movie), rating) }.join(predictions)
    val diff = join_w_trueRating.map(line => abs(line._2._1 - line._2._2)).collect()
    val squareSum = join_w_trueRating.map(line => (1, (pow((line._2._1 - line._2._2), 2), 1))).reduceByKey {
      case (l1, l2) => (l1._1 + l2._1, l1._2 + l2._2)
    }.collect()
    val RMSE = sqrt(squareSum(0)._2._1 / squareSum(0)._2._2)
    val pred = predictions.map(line => (line._1._1, line._1._2, line._2))
    return (pred, diff, RMSE)
  }

  def write_result(filePath: String, prediction: RDD[(Int, Int, Double)], difference: Array[Double], rmse: Double): Unit = {
    val count = Array(0, 0, 0, 0, 0)
    for (ele <- difference) {
      if (ele < 1.0) {
        count(0) += 1
      } else if (ele >= 1.0 && ele < 2.0) {
        count(1) += 1
      } else if (ele >= 2.0 && ele < 3.0) {
        count(2) += 1
      } else if (ele >= 3.0 && ele < 4.0) {
        count(3) += 1
      } else {
        count(4) += 1
      }
    }
    println(">=0 and <1: " + count(0).toString)
    println(">=1 and <2: " + count(1).toString)
    println(">=2 and <3: " + count(2).toString)
    println(">=3 and <4: " + count(3).toString)
    println(">=4: " + count(4).toString)
    println("RMSE = " + rmse.toString)

    val file = new File(filePath)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write("UserId, MovieId, Pred_rating\n")
    val result = prediction.collect()
    for (record <- result) {
      val str = record._1.toString + ", " + record._2.toString + ", " + record._3.toString + "\n"
      bw.write(str)
    }
    bw.close()

  }
}