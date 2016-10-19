local nn = require 'nn'



-- params
inputSize = 4000
outputSize = 11
featureSize = 4000

-- Create encoder
encoder = nn.Sequential()
encoder:add(nn.View(-1, featureSize))
encoder:add(nn.Linear(featureSize, outputSize))
encoder:add(nn.ReLU(true))

-- Create decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize, featureSize))
decoder:add(nn.Sigmoid(true))
decoder:add(nn.View(featureSize))

-- Create autoencoder
autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)


return autoencoder
