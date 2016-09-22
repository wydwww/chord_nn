package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import com.intel.webscaleml.nn.example.chord._

import scala.collection.mutable.ArrayBuffer
//import com.intel.webscaleml.nn.example.MNIST._
import com.intel.webscaleml.nn.example.Utils._

import scala.util.Random

object chordLocal {

  def main(args : Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {train(params)}}.getOrElse{sys.exit(1)}
  }

  def shuffle[T](data : Array[T]) = {
    var i = 0
    while(i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  def train(params : Utils.Params) = {
    val folder = params.folder
    val (trainData, testData) = loadFile()
    println("train data length = " + trainData.length)
    println("test data length = " + testData.length)
    val module = getModule()
    val optm = getOptimMethod(params.masterOptM)
    val critrion = new ClassNLLCriterion[Double]()
    val (w, g) = module.getParameters()
    var e = 0
    val config = params.masterConfig.clone()
    config("dampening") = 0.0
//    config("learningRate") = 0.01
//    config("learningRateDecay") = 5e-7
    var wallClockTime = 0L
//    val (mean, std) = computeMeanStd(trainData)
//    println(s"mean is $mean std is $std")
    val input = torch.Tensor[Double]()
    val target = torch.Tensor[Double]()
    println("start!")
    while(e < 20) {
//      while(e < config.get[Int]("epoch").get) {
//      println("epoch: " + e)
//      shuffle(trainData)
      var trainLoss = 0.0
      var i = 0
      var c = 0
      while(i < trainData.length) {
        val start = System.nanoTime()
        val batch = math.min(10, trainData.length - i)
        val buffer = new Array[Array[Double]](batch)
        var j = 0
        while(j < buffer.length) {
          buffer(j) = trainData(i + j)
          j += 1
        }

        toTensor(buffer, input, target)
        module.zeroGradParameters()
        val output = module.forward(input)
        val loss = critrion.forward(output, target)
        val gradOutput = critrion.backward(output, target)
        module.backward(input, gradOutput)
        optm.optimize(_ => (loss, g), w, config, config)
        val end = System.nanoTime()
        trainLoss += loss
        wallClockTime += end - start
        println(s"[Wall Clock ${wallClockTime.toDouble / 1e9}s][epoch $e][iteration $c $i/${trainData.length}] Train time is ${(end - start) / 1e9}seconds. loss is $loss Throughput is ${buffer.length.toDouble / (end - start) * 1e9} records / second")

        i += buffer.length
        c += 1
      }
      println(s"[epoch $e][Wall Clock ${wallClockTime.toDouble / 1e9}s] Training Loss is ${trainLoss / c}")

      // eval
      var k = 0
      var correct = 0
      var count = 0
      var testLoss = 0.0
      val buffer1 = torch.Tensor[Double]()
      val buffer2 = torch.Tensor[Double]()
      while(k < testData.length) {
        val (input, target) = toTensor(Array(testData(k)), buffer1, buffer2)
        val output = module.forward(input)
        testLoss += critrion.forward(output, target)
        val (curCorrect, curCount) = EvaluateMethods.calcAccuracy(output, target)
        correct += curCorrect
        count += curCount
        k += 1
      }
      println(correct)
      println(count)
      println(s"[Wall Clock ${wallClockTime.toDouble / 1e9}s] Test Loss is ${testLoss / k} Accuracy is ${correct.toDouble / count}")

      e += 1
    }
  }



}
