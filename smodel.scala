package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}

import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.json.Parser
import scala.io.Source
import scala.util.Random

object chord extends App {

  val trSize = 9600
  val teSize = 2393
  val classes = Map(("A4", 1), ("B3", 2), ("B4", 3), ("C3", 4), ("C3A4", 5), ("C3B4", 6), ("C3C4", 7), ("C3D4", 8),
    ("C3E4", 9), ("C3F4", 10), ("C3G4", 11), ("C4", 12), ("C4A4", 13), ("C4AS4", 14), ("C4B4", 15), ("C4CS4", 16),
    ("C4D4", 17), ("C4DS4", 18), ("C4E4", 19), ("C4E4G4", 20), ("C4F4", 21), ("C4FS4", 22), ("C4G4", 23),
    ("C4GS4", 24), ("C5", 25), ("D4", 26), ("D4E4F4", 27), ("E4", 28), ("F4", 29), ("G4", 30))
  loadFile()

  //  def toTensor(input : Tensor[Double], target : Tensor[Double]) : (Tensor[Double]) = {
  //
  //  }
  //  def toTensor(inputs : Seq[Array[Byte]], input : Tensor[Double], target : Tensor[Double]) : (Tensor[Double]) = {
  //
  //  }

  def shuffle[T](data : ArrayBuffer[T]) = {
    var i = 0
    while(i < data.length) {
      val exchange = i + Random.nextInt(data.length - i)
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
  }

  def loadFile() = {
    val filesHere = new java.io.File("/Users/intel/Downloads/").list.filter(_.endsWith("txt"))
    val labelBuffer = new ArrayBuffer[Int]()
    var featureBuffer = new ArrayBuffer[Double]()
    val pattern = "[A-Z]?[0-9]?[A-Z]?[A-Z]?[0-9]?[A-Z]?[0-9]?".r
    for (file <- filesHere) {
      labelBuffer += classes(pattern.findFirstIn(file).get)
      //      labelBuffer.foreach(f=>println(f))
      for (line <- Source.fromFile("/Users/intel/Downloads/" + file).getLines()) {
        featureBuffer += line.toDouble
      }
    }
    //      println(featureBuffer(3))
    //      featureBuffer.foreach(f=>println(f))

    shuffle(labelBuffer)
    shuffle(featureBuffer)

    val trainData = new Array[Array[Double]](trSize)
    var i = 0
    while (i < trSize) {
      val s = new Array[Double](trSize * 4000 + 1)
      s(0) = labelBuffer(i)
      var y = 0
      while (y < trSize) {
        var x = 0
        while (x < 4000) {
          s(1 + x + y * 4000) = featureBuffer(x + y * 4000)
          x += 1
        }
        y += 1
      }
      trainData(i) = s
      i += 1
    }

    val testData = new Array[Array[Double]](teSize)
    var j = 0
    while (j < teSize) {
      val s = new Array[Double](teSize * 4000 + 1)
      s(0) = labelBuffer(j + trSize)
      var y = 0
      while (y < teSize) {
        var x = 0
        while (x < 4000) {
          s(1 + x + y * 4000) = featureBuffer(x + (y + trSize) * 4000)
          x += 1
        }
        y += 1
      }
      testData(j) = s
      j += 1
    }
  }
}
