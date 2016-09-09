require 'torch'
require 'nn'
require 'optim'

dofile 'data.lua'

-- parameter table for grid search
param = {}
param.learningRate = {0.01, 0.001}
param.HUs = {2000, 2500, 3000, 3500, 4000, 4500, 5000}
param.batchSize = {}

-- 30-classes classification problem
classes = {'A4', 'B3', 'B4', 'C3', 'C3A4', 'C3B4', 'C3C4', 'C3D4', 'C3E4', 'C3F4', 'C3G4', 'C4', 'C4A4', 'C4AS4', 'C4B4', 'C4CS4', 'C4D4', 'C4DS4', 'C4E4', 'C4E4G4', 'C4F4', 'C4FS4', 'C4G4', 'C4GS4', 'C5', 'D4', 'D4E4F4', 'E4', 'F4', 'G4'}
trSize = 9600
teSize = 2390 -- max:2393 integer multiple of batchSize

-- Log results to files
trainLogger = optim.Logger(paths.concat('train.log'))
testLogger = optim.Logger(paths.concat('test.log'))

-- grid search prototype
for i,v in ipairs(param.learningRate) do 
   lr = v
   for i,v in ipairs(param.HUs) do
      hu = v
      --trainLogger = optim.Logger(paths.concat('train_'..'lr'..lr..'_HUs'..hu..'.log'))
      --testLogger = optim.Logger(paths.concat('test_'..'lr'..lr..'_HUs'..hu..'.log'))
      print(lr..', '..hu)
   end
end

-- SGD
torch.setdefaulttensortype('torch.FloatTensor')
batchSize = 10
learningRate = 0.01
momentum = 0.5
learningRateDecay = 5e-7

-- creat a multi-layer perceptron
mlp = nn.Sequential()
inputs = 4000 -- the number of input dimensions
outputs = 30 -- number of classifications
HUs = 2000 -- hidden units
mlp:add( nn.Linear(inputs, HUs) )
mlp:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
mlp:add( nn.Linear(HUs, outputs) )
mlp:add( nn.LogSoftMax() )

-- retrieve parameters and gradients
parameters,gradParameters = mlp:getParameters()

criterion = nn.ClassNLLCriterion()

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

-- train function
function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   
   -- do one epoch
   print('<trainer> on training set:')
   print(<trainer> online epoch #  .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   print('[hiddenUnits = ' .. HUs .. ']')
   print('[learningRate = ' .. learningRate .. ']')
   print('[momentum = ' .. momentum .. ']') 
   
   shuffle = torch.randperm(9600)
   for t = 1,trainData:size(),batchSize do
      -- create mini batch
      local inputs = torch.Tensor(batchSize,4000)
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         inputs[k] = trainData.data[i]
         targets[k] = trainData.labels[i]
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = mlp:forward(inputs)
         f = criterion:forward(outputs, targets)
         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         mlp:backward(inputs, df_do)

         -- return f and df/dX
         return f,gradParameters
      end

      -- Perform SGD step:
      sgdState = sgdState or {
         learningRate = learningRate,
         momentum = momentum,
         learningRateDecay = learningRateDecay,
         weightDecay = 0
      }
      optim.sgd(feval, parameters, sgdState)
      -- disp progress
      xlua.progress(t, trainData:size())
      
      loss = f
   end
   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print('[train loss: ]' .. loss)
 
   -- log train result
   trainLogger:add{['% tloss'] = loss}
   
   -- save/log current net
   --local filename = './model.net'
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   --print('==> saving model to '..filename)
   --torch.save(filename, mlp)

   epoch = epoch + 1

end

-- test function
function test()
   -- local vars
   local time = sys.clock()
   correct = 0
   -- test over given dataset
   for t = 1, teSize, batchSize do
      -- disp progress
      xlua.progress(t, teSize)

      -- create mini batch
      local inputs = torch.Tensor(batchSize,4000)
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1, teSize) do
         -- load new sample
         inputs[k] = testData.data[i]
         targets[k] = testData.labels[i]
         k = k + 1
      end

      -- test samples
      local preds = mlp:forward(inputs)
      -- vloss: validation loss
      local validationLoss = criterion:forward(preds, targets)
      vloss = validationLoss
      local confidences, indices = torch.sort(preds, true)  -- true means sort in descending order
      for j = 1, targets:size(1) do
         if targets[j] == indices[j][1] then
            correct = correct + 1
         end
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / teSize
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
   print('correct count: ' .. correct)
   print('accuracy: ' .. (correct/teSize*100) .. '%')

   testLogger:add{['% vloss'] = vloss, ['% correct'] = correct, ['% accuracy'] = correct/teSize*100}

end

-- start!
while not epoch or epoch<16 do
   train()
   test()

end

-- plot the log   
trainLogger:style{['% tloss'] = '+-'}
testLogger:style{['% accuracy'] = '+-', ['% vloss'] = '+-'}
trainLogger:plot()
testLogger:plot()

