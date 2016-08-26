require 'torch'
require 'nn'
require 'paths'

trSize = 9600
teSize = 2393

if paths.filep('train.t7') and paths.filep('test.t7') then

   print('==> loading previously generated dataset:')

   trainData = torch.load('train.t7')
   testData = torch.load('test.t7')

else

   classes = {A4 = 1, B3 = 2, B4 = 3, C3 = 4, C3A4 = 5, C3B4 = 6, C3C4 = 7, C3D4 = 8, C3E4 = 9, C3F4 = 10, C3G4 = 11, C4 = 12, C4A4 = 13, C4AS4 = 14, C4B4 = 15, C4CS4 = 16, C4D4 = 17, C4DS4 = 18, C4E4 = 19, C4E4G4 = 20, C4F4 = 21, C4FS4 = 22, C4G4 = 23, C4GS4 = 24, C5 = 25, D4 = 26, D4E4F4 = 27, E4 = 28, F4 = 29, G4 = 30}

   print('==> creating a new dataset from raw files:')
   
   t = {}
   for f in paths.files("/home/arda/yiding/segments1/data") do
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
      local f = assert(io.open("/home/arda/yiding/segments1/data/"..file_name, "r"))
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
      
      print('Loading train data '..i..'/9600 : '..t[i])
      trainData.data[i] = readfile(t[i])
      trainData.labels[i] = classes[string.match(t[i],"%u?%d?%u?%u?%d?%u?%d?")]
   
   end
   
   for i = 1, teSize do
   
      print('Loading test data '..i..'/'..teSize..' : '..t[i+trSize])
      testData.data[i] = readfile(t[i+trSize])
      testData.labels[i] = classes[string.match(t[i+trSize],"%u?%d?%u?%u?%d?%u?%d?")]
   
   end
   
   -- save datasets to files
   torch.save('train.t7', trainData)
   torch.save('test.t7', testData)

end

trainData.size = function() return trSize end
testData.size = function() return teSize end

-- export
return {
   trainData = trainData,
   testData = testData,
}

