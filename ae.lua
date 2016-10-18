local nn = require 'nn'



-- params
inputSize = 4000
outputSize = 11
featureSize = 4000
-- encoder
-- encoder = nn.Sequential()
-- encoder:add(nn.Linear(inputSize,outputSize))
-- encoder:add(nn.Tanh())
-- encoder:add(nn.Diag(outputSize))

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))

-- -- Create encoder
encoder = nn.Sequential()
encoder:add(nn.View(-1, featureSize))
encoder:add(nn.Linear(featureSize, 32))
encoder:add(nn.ReLU(true))

-- -- Create decoder
-- self.decoder = nn.Sequential()
-- self.decoder:add(nn.Linear(32, featureSize))
-- self.decoder:add(nn.Sigmoid(true))
-- self.decoder:add(nn.View(featureSize))

-- Create autoencoder
autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)


return autoencoder
