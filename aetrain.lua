local optim = require 'optim'
local gnuplot = require 'gnuplot'
local image = require 'image'
dofile 'real_data.lua'

-- Set up Torch
print('Setting up')
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

classes = {'1','2','3','4','5','6','7','8','9','10','11'}
trSize = 1540
teSize = 390

-- load train and test data
setmetatable(trainData, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
)

setmetatable(testData,
    {__index = function(t, i)
                    return {t.data[i], t.labels[i]}
                end}
)


-- Choose model to train
local cmd = torch.CmdLine()
cmd:option('-learningRate', 0.001, 'Learning rate')
cmd:option('-optimiser', 'adam', 'Optimiser')
local epochs = 10
local learningRate = 0.001
local batchSize = 10
local model = require 'ae'