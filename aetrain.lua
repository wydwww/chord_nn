local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

-- Load train and test data


-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
local epochs = 10
local learningRate = 0.001
local batchSize = 10
local model = require 'ae'