package com.intel.webscaleml.nn.example

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}
import java.io._

import com.intel.webscaleml.nn.nn._
import com.intel.webscaleml.nn.optim._
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}

import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.json.Parser
import scala.io.Source
import scala.util.Random

object chord extends App {

//  val trSize = 9600
//  val teSize = 2393
  val inputSize = 4000
  val outputs = 10
  val HUs = 2000

  val classes = Map(("A4", 1), ("B3", 2), ("B4", 3), ("C3", 4), ("C4", 5), ("C5", 6), ("D4", 7), ("E4", 8), ("F4", 9), ("G4", 10))
//  val classes = Map(("A4", 1), ("B3", 2), ("B4", 3), ("C3", 4), ("C3A4", 5), ("C3B4", 6), ("C3C4", 7), ("C3D4", 8),
//    ("C3E4", 9), ("C3F4", 10), ("C3G4", 11), ("C4", 12), ("C4A4", 13), ("C4AS4", 14), ("C4B4", 15), ("C4CS4", 16),
//    ("C4D4", 17), ("C4DS4", 18), ("C4E4", 19), ("C4E4G4", 20), ("C4F4", 21), ("C4FS4", 22), ("C4G4", 23),
//    ("C4GS4", 24), ("C5", 25), ("D4", 26), ("D4E4F4", 27), ("E4", 28), ("F4", 29), ("G4", 30))
  loadFile()

//  def shuffleBuffer[T](data : ArrayBuffer[T]) = {
//    var i = 0
//    while(i < data.length) {
//      val exchange = i + Random.nextInt(data.length - i)
//      val tmp = data(exchange)
////      val tmp1 = label(exchange)
//      data(exchange) = data(i)
////      label(exchange) = label(i)
//      data(i) = tmp
////      label(i) = tmp1
//      i += 1
//    }
//  }

  def loadFile():(Array[Array[Double]], Array[Array[Double]]) = {
//    val filesHere = new java.io.File("/Users/intel/WebScaleML/algorithms/src/main/scala/com/intel/webscaleml/nn/example/yiding/").list.filter(_.endsWith("asc"))
//    println("There are " + filesHere.size + "files.")
    val labelBuffer = new ArrayBuffer[Int]()
    val featureBuffer = Array.ofDim[Double](7998,4000)//new Array[Array[Double]](7998)
//    val pattern = "[A-Z]?[0-9]?[A-Z]?[A-Z]?[0-9]?[A-Z]?[0-9]?".r
//    var count1 = 1
//    var count2 = 1
    val classes = Map(("A4", 1), ("B3", 2), ("B4", 3), ("C3", 4), ("C4", 5), ("C5", 6), ("D4", 7), ("E4", 8), ("F4", 9), ("G4", 10))
    val trSize = 6400
    val teSize = 1590

    val filename = "/Users/intel/data.asc"
    var count = 0
    println("loading file")
    for (line <- Source.fromFile(filename).getLines) {
//      println(line)

      val a = line.split(" ")
      labelBuffer += classes(a(0))

//      println(a(1))
      var ii = 0
      while (ii < 4000) {
        featureBuffer(count)(ii) = a(ii+1).toDouble
        ii += 1
      }
      println("count" + count)
      count += 1
    }

//    for ( i <- 1 to 10){
//      println(featureBuffer(i))
//    }
//    for ( i <- 7000 to 7010){
//      println(labelBuffer(i))
//    }

//    for (file <- filesHere) {
//      println("Loading data: " + count1 + " / " + filesHere.size)
//      labelBuffer += classes(pattern.findFirstIn(file).get)
//      println(labelBuffer(count1-1))
//      count1 += 1
////      println(file)
//      //      labelBuffer.foreach(f=>println(f))
//      for (line <- Source.fromFile("/Users/intel/WebScaleML/algorithms/src/main/scala/com/intel/webscaleml/nn/example/yiding/" + file).getLines()) {
//        featureBuffer += line.toDouble
////        println(line)
//      }
//    }

//      println(featureBuffer(3))
//      featureBuffer.foreach(f=>println(f))


//    shuffleBuffer(featureBuffer)
//    shuffleBuffer(labelBuffer)

    val trainData = new Array[Array[Double]](trSize)
    var i = 0
    while (i < trSize) {
      val s = new Array[Double](4000 + 1)
      s(0) = labelBuffer(i)
//      println("s(0): "+s(0))
      var x = 0
      while (x < 4000) {
        s(1 + x) = featureBuffer(i)(x)
        x += 1
      }
//      println("i: " + i + ", s.length: " + s.length)
      trainData(i) = s
      i += 1
    }
    println("trainData.length = " + trainData.length)

    val testData = new Array[Array[Double]](teSize)
    var j = 0
    while (j < teSize) {
      val ss = new Array[Double](4000 + 1)
      ss(0) = labelBuffer(j + trSize)
      var x = 0
      while (x < 4000) {
        ss(1 + x) = featureBuffer(j + trSize)(x)
        x += 1
      }
//      println("ss.length: " + ss.length)
      testData(j) = ss
      j += 1
    }
    println("testData.length = " + testData.length)

    (trainData, testData)
  }

  def getModule() : Module[Double] = {
//    val inputSize = 4000
//    val outputs = 10
//    val HUs = 2000
    val mlp = new Sequential[Double]

    mlp.add(new Reshape(Array(4000)))
    mlp.add(new Linear(4000, 2000))
    mlp.add(new Tanh)
    mlp.add(new Linear(2000, 10))
    mlp.add(new LogSoftMax)
    mlp

    }

  def toTensor(inputs : Seq[Array[Double]], input : Tensor[Double], target : Tensor[Double]) : (Tensor[Double], Tensor[Double]) = {
    val inputSize = 4000
    val size = inputs.size //size = 10
//    println("toTensor inputs.size = " + size)
    input.resize(Array(size, inputSize))
//    println("toTensor input.size(1) = " + input.size(1))

    target.resize(Array(size))
    var i = 0
    while(i < size) {
      val img = inputs(i)
      var j = 0
      while(j < inputSize) {
        input.setValue(i + 1, j + 1, img(j + 1))
        j += 1
      }
      target.setValue(i + 1, img(0))
      i += 1
    }
//    println("input size = " + input.size)
//    println("target size = " + target.size)
    (input, target)
  }
}
