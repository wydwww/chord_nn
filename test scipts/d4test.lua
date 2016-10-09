require 'torch'
require 'nn'
require 'optim'

dofile 'singlekey_data.lua'

-- parameter table for grid search
param = {}
param.learningRate = {0.01, 0.001}
param.HUs = {2000, 2500, 3000, 3500, 4000, 4500, 5000}
param.batchSize = {}

-- 30-classes classification problem
classes = {'A4', 'B3', 'B4', 'C3', 'C4', 'C5', 'D4', 'E4', 'F4', 'G4'}
trSize = 6400
teSize = 1590 -- max:1598 integer multiple of batchSize

-- Log results to files
trainLogger = optim.Logger(paths.concat('singletrain.log'))
testLogger = optim.Logger(paths.concat('singletest.log'))

-- grid search prototype
--for i,v in ipairs(param.learningRate) do 
--   lr = v
--   for i,v in ipairs(param.HUs) do
--      hu = v
--      --trainLogger = optim.Logger(paths.concat('train_'..'lr'..lr..'_HUs'..hu..'.log'))
--      --testLogger = optim.Logger(paths.concat('test_'..'lr'..lr..'_HUs'..hu..'.log'))
--      print(lr..', '..hu)
--   end
--end

-- SGD
torch.setdefaulttensortype('torch.FloatTensor')
batchSize = 10
learningRate = 0.01
momentum = 0
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
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   print('[hiddenUnits = ' .. HUs .. ']')
   print('[learningRate = ' .. learningRate .. ']')
   print('[momentum = ' .. momentum .. ']') 
   
   shuffle = torch.randperm(6400)
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


tt = {}
for f in paths.files("/home/arda/yiding/c3c4d4asc") do
   table.insert(tt,f)
end
table.sort(tt)
table.remove(tt,1)
table.remove(tt,1)

function mysplit(inputstr, sep)
       if sep == nil then
               sep = "%s"
       end
       local t = {}
       local i = 1
       for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
               t[i] = str
               i = i + 1
       end
       return t
end

function readfile(file_name)
      local f = assert(io.open("/home/arda/yiding/c3c4d4asc/"..file_name, "r"))
      local t = f:read("*all")
      f:close()
      y = mysplit(t)
      x = torch.Tensor(y):resize(1,4000)
      return x
end

math.randomseed( os.time() )

local function shuffleTable( t )
       local rand = math.random
       assert( t, "shuffleTable() expected a table, got nil" )
       local iterations = #t
       local j

       for i = iterations, 2, -1 do
           j = rand(i)
           t[i], t[j] = t[j], t[i]
       end
end

shuffleTable(tt)
d4size = 614
classes = {A4 = 1, B3 = 2, B4 = 3, C3 = 4, C4 = 12, C5 = 25, D4 = 26, E4 = 28, F4 = 29, G4 = 30}
d4data = {
   data = torch.Tensor(d4size, 4000),
   labels = torch.Tensor(d4size),
   size = function() return d4size end
}

for i = 1, d4size do

  print('Loading d4 data '..i..'/307 : '..tt[i])
  d4data.data[i] = readfile(tt[i])
  print(classes[string.match(tt[i],"%u?%d?%u?%u?%d?%u?%d?")])
  d4data.labels[i] = classes[string.match(tt[i],"%u?%d?%u?%u?%d?%u?%d?")]

end


d4size = 614
function d4test()

   d4correct = 0
   for t = 1, d4size, 1 do
      xlua.progress(t, d4size)
      local inputs = torch.Tensor(1,4000)
      local targets = torch.Tensor(1)
      local k = 1
      for i = t,math.min(t, d4size) do
	      inputs[k] = d4data.data[i]
	      targets[k] = d4data.labels[i]
	      k = k + 1
      end

      local preds = mlp:forward(inputs)

      local validationLoss = criterion:forward(preds, targets)
      local confidences, indices = torch.sort(preds, true)
      for j = 1, targets:size(1) do
	      if targets[j] == indices[j][1] then
		      d4correct = d4correct + 1
              end
      end
   end

   print('d4 correct count: ' .. d4correct)
   print('d4 accuracy: ' .. (d4correct/d4size*100) .. '%')

end

-- start!
while not epoch or epoch<1 do
   train()
   test()

end
d4test()
-- plot the log   
trainLogger:style{['% tloss'] = '+-'}
testLogger:style{['% accuracy'] = '+-', ['% vloss'] = '+-'}
trainLogger:plot()
testLogger:plot()

