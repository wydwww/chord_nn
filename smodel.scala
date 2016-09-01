package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Paths, Files}

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scopt.OptionParser
import com.intel.webscaleml.nn.example.MNIST._
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
    val trainData =
  }




}
