local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
dofile 'real_data.lua'

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
local XTrain = trainData.data:float():div(255)
local XTest = testData.data:float():div(255)
-- print(trainData.labels)
-- print(trainData.size)
trSize = 1540
teSize = 390

-- load train and test data
-- setmetatable(trainData, 
--     {__index = function(t, i) 
--                     return {t.data[i], t.labels[i]} 
--                 end}
-- )

-- setmetatable(testData,
--     {__index = function(t, i)
--                     return {t.data[i], t.labels[i]}
--                 end}
-- )


-- Choose model to train
local optimiser = 'sgd'
local epochs = 10
local learningRate = 0.001
local batchSize = 10
-- local model = require 'ae'
-- model:createAutoencoder(XTrain)
-- print(model[0])
local autoencoder = require 'ae'

-- Get parameters
local theta, gradTheta = autoencoder:getParameters()
local thetaAdv, gradThetaAdv

-- Create loss
local criterion = nn.BCECriterion()

-- Create optimiser function evaluation
local x -- Minibatch
local feval = function(params)
  if theta ~= params then
    theta:copy(params)
  end
  -- Zero gradients
  gradTheta:zero()
  
  -- Reconstruction phase
  -- Forward propagation
  local xHat = autoencoder:forward(x) -- Reconstruction
  local loss = criterion:forward(xHat, x)
  -- Backpropagation
  local gradLoss = criterion:backward(xHat, x)
  autoencoder:backward(x, gradLoss)

  return loss, gradTheta
end

local advFeval = function(params)
  if thetaAdv ~= params then
    thetaAdv:copy(params)
  end

  return advLoss, gradThetaAdv
end

-- Train
print('Training')
autoencoder:training()
local optimParams = {learningRate}
local advOptimParams = {learningRate}
local __, loss
local losses, advLosses = {}, {}

for epoch = 1, epochs do
  print('Epoch ' .. epoch .. '/' .. epochs)
  for n = 1, trSize, batchSize do
    -- Get minibatch
    x = XTrain:narrow(1, n, batchSize)

    -- Optimise
    __, loss = optim[optimiser](feval, theta, optimParams)
    losses[#losses + 1] = loss[1]

    -- Train adversary
    if opt.model == 'AAE' then
      __, loss = optim[optimiser](advFeval, thetaAdv, advOptimParams)     
      advLosses[#advLosses + 1] = loss[1]
    end
  end

  -- Plot training curve(s)
  local plots = {{'Autoencoder', torch.linspace(1, #losses, #losses), torch.Tensor(losses), '-'}}
  
  gnuplot.pngfigure('Training.png')
  gnuplot.plot(table.unpack(plots))
  gnuplot.ylabel('Loss')
  gnuplot.xlabel('Batch #')
  gnuplot.plotflush()
end

-- Test
print('Testing')
x = XTest:narrow(1, 1, 10)
local xHat

autoencoder:evaluate()
xHat = autoencoder:forward(x)

-- Plot reconstructions
image.save('Reconstructions.png', torch.cat(image.toDisplayTensor(x, 2, 10), image.toDisplayTensor(xHat, 2, 10), 1))
