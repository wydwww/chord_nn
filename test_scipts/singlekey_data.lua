require 'torch'
require 'nn'
require 'paths'

trSize = 6400
teSize = 1590 --1598

if paths.filep('singletrain.t7') and paths.filep('singletest.t7') then

   print('==> loading previously generated dataset:')

   trainData = torch.load('singletrain.t7')
   testData = torch.load('singletest.t7')

else

   classes = {A4 = 1, B3 = 2, B4 = 3, C3 = 4, C4 = 12, C5 = 25, D4 = 26, E4 = 28, F4 = 29, G4 = 30}

   print('==> creating a new dataset from raw files:')
   
   t = {}
   for f in paths.files("/home/arda/yiding/singleData") do
      table.insert(t,f)
   end
   -- remove blank names
   table.sort(t)
   table.remove(t,1)
   table.remove(t,1)
   --table.sort(t, function(a, b) return a > b end)
   
   -- split asc file
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
   
   -- read data from file
   function readfile(file_name)
      local f = assert(io.open("/home/arda/yiding/singleData/"..file_name, "r"))
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
   
   shuffleTable(t)
   shuffleTable(t)
   shuffleTable(t)

   -- build train and test datasets
   trainData = {
      data = torch.Tensor(trSize, 4000),
      labels = torch.Tensor(trSize),
      size = function() return trSize end
   }
   
   testData = {
      data = torch.Tensor(teSize, 4000),
      labels = torch.Tensor(teSize),
      size = function() return teSize end
   }
   
   -- load data and labels
   for i = 1, trSize do
      
      print('Loading train data '..i..'/6400 : '..t[i])
      trainData.data[i] = readfile(t[i])
      trainData.labels[i] = classes[string.match(t[i],"%u?%d?%u?%u?%d?%u?%d?")]
   
   end
   
   for i = 1, teSize do
   
      print('Loading test data '..i..'/'..teSize..' : '..t[i+trSize])
      testData.data[i] = readfile(t[i+trSize])
      testData.labels[i] = classes[string.match(t[i+trSize],"%u?%d?%u?%u?%d?%u?%d?")]
   
   end
   
   -- save datasets to files
   torch.save('singletrain.t7', trainData)
   torch.save('singletest.t7', testData)

end

trainData.size = function() return trSize end
testData.size = function() return teSize end

-- export
return {
   trainData = trainData,
   testData = testData,
}

