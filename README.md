# chord_nn
A neural network with torch for chord recognition
## music_process.py
Split the wav files to several files by blank parts
## time2freq.py
Do fft and cut the frequency up to 4000Hz
## delete_nan.py
Delete null data
## data.lua
Load data and build datasets
## model.lua
Build a neural network, then train and test
## smodel.scala
Model in scala
