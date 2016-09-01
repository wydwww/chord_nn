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

  val trSize = 9600
  val teSize = 2393

  val classes = Map(("A4", 1), ("B3", 2), ("B4", 3), ("C3", 4), ("C3A4", 5), ("C3B4", 6), ("C3C4", 7), ("C3D4", 8), ("C3E4", 9), ("C3F4", 10), ("C3G4", 11), ("C4", 12), ("C4A4", 13), ("C4AS4", 14), ("C4B4", 15), ("C4CS4", 16), ("C4D4", 17), ("C4DS4", 18), ("C4E4", 19), ("C4E4G4", 20), ("C4F4", 21), ("C4FS4", 22), ("C4G4", 23), ("C4GS4", 24), ("C5", 25), ("D4", 26), ("D4E4F4", 27), ("E4", 28), ("F4", 29), ("G4", 30))

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  def shuffle[T](data: Array[T]) = {
    var i = 0
    while (i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  def train(params: Utils.Params) = {
    val folder = params.folder
    var traindata =
    val trainData =
  }


}
