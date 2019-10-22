require 'image'
require 'optim'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')
-- load the MNIST dataset
--local dl = require 'dataload'

--train, valid, test = dl.loadMNIST()
--indices = torch.LongTensor(train:size())
--for i=1,train:size() do
--   indices[i]=i
--end
--train_inputs, train_targets = train:index(indices)
 

--test_inputs, test_targets = test:index(indices[{{1,10000}}])


--valid_inputs, valid_targets = valid:index(indices[{{1,10000}}])

train_inputs = torch.FloatTensor(100,1,154,154)
for i = 1,100 do
train_inputs[i] = image.load('enronData/data'.. i ..'.png')
end

--set data for a special number
--num = 4
--s =torch.sum(torch.eq(train_targets,num))


--s = train:size()
s = 100

--train_inputs = torch.FloatTensor(s,1,28,28)
--local count = 1
--for i=1,50000 do
--   if train_targets[i] == num then
--     train_inputs[count] = TrainNew[i]
--      count = count +1 
--   end
--end
	
--define a network architecture 

d_mlp = nn.Sequential()
d_mlp:add(nn.View(154*154))
d_mlp:add(nn.Linear(154*154,500))
--d_mlp:add(nn.BatchNormalization(1000))
d_mlp:add(nn.ReLU())
--d_mlp:add(nn.Dropout(0.4))
--d_mlp:add(nn.Linear(200,200))
--d_mlp:add(nn.ReLU())
--d_mlp:add(nn.Dropout(0.4))
d_mlp:add(nn.Linear(500,1))
d_mlp:add(nn.Sigmoid())




g_mlp = nn.Sequential()
g_mlp:add(nn.View(100))
g_mlp:add(nn.Linear(100,1000))
g_mlp:add(nn.BatchNormalization(1000))
g_mlp:add(nn.ReLU())
g_mlp:add(nn.Linear(1000,1000))
g_mlp:add(nn.BatchNormalization(1000))
g_mlp:add(nn.ReLU())
g_mlp:add(nn.Linear(1000,154*154))
g_mlp:add(nn.Sigmoid())




-----------------------------------optimise the learning rate-----------
 
optimState = {
        learningRate = 0.01,
        learningRateDecay = 0.0,
        beta1 = 0.9,
        beta2 = 0.0,
        epsilon = 5e-4
}
optimState2 = {
        learningRate = 0.01,
        learningRateDecay = 0.0,
        beta1 = 0.9,
        beta2 = 0.0,
        epsilon = 5e-4
}
parameters1, gradParameters1 = d_mlp:getParameters() 
parameters2, gradParameters2 = g_mlp:getParameters() 
 



----------------------------------optimize all layers parameters at once------------
local function w_init(net)
   for k,v in pairs(net:findModules'nn.Linear') do
   v.weight:normal(0,1)
   if v.bias~=nil then 
     v.bias:normal(0,1)
   end
   end
end

--w_init(g_mlp)

--train the network

--print(nn.SoftMax():forward(mlp:forward(train_inputs[1])))

--lr=0.01
for j = 1,100000 do
 

   --lr=lr*0.99
   --create dataset
--------------------------------------------------
bt=10
   g_inputs=torch.Tensor(s,100):normal(0,1)
   inputs=torch.cat(train_inputs,g_mlp:forward(g_inputs):view(g_inputs:size(1),1,154,154),1)
   targets=torch.cat(torch.ones(s),torch.zeros(g_inputs:size(1)))
   --shuffling the data
   ShuffleIndex = torch.randperm(inputs:size(1)):long()
   inputs = inputs:index(1,ShuffleIndex)
   targets = targets:index(1,ShuffleIndex)
   g_targets=torch.ones(g_inputs:size(1))
   d_mlp:training()   
   for i=1,inputs:size(1)/bt do
        xinput=inputs[{{(i-1)*bt+1,i*bt},{1},{},{}}]
	yinput=targets[{{(i-1)*bt+1,i*bt}}]

local feval1 = function(x)
  d_mlp:zeroGradParameters()	
  criterion = nn.BCECriterion()
  pred = d_mlp:forward(xinput)
  out = criterion:forward(pred, yinput)
  err = criterion:backward(pred, yinput)
  d_mlp:backward(xinput, err)
  return 0, gradParameters1
end

	optim.adam(feval1,parameters1, optimState)
	 
	
   end
pred = d_mlp:forward(inputs)
print('after traing D '..torch.sum(torch.eq(targets:byte(),pred:ge(0.5)))/pred:size(1) )

bt=10	

--lr=0.01
   for i=1,g_inputs:size(1)/bt do 
        xinput2=g_inputs[{{(i-1)*bt+1,i*bt}}]
	yinput2=g_targets[{{(i-1)*bt+1,i*bt}}]
feval2 = function(x2)
  if x2~=parameters2 then
    x2:copy(parameters2)
  end
  g_mlp:zeroGradParameters()
  criterion = nn.BCECriterion()
  d_inp=g_mlp:forward(xinput2)
  pred = d_mlp:forward(d_inp)
  out = criterion:forward(pred, yinput2)
  err = criterion:backward(pred, yinput2)
  grads=d_mlp:backward(d_inp, err)
  g_mlp:backward(xinput2, grads)
  return 0, gradParameters2
end
	optim.adam(feval2, parameters2, optimState2)
	 
   end

criterion = nn.BCECriterion()
d_mlp:evaluate()
inputs=torch.cat(train_inputs,g_mlp:forward(g_inputs):view(g_inputs:size(1),1,154,154),1)
pred = d_mlp:forward(inputs)
y=torch.cat(torch.ones(s),torch.zeros(g_inputs:size(1)))
out = criterion:forward(pred, y)
err = criterion:backward(pred, y)
--print (err)
print( torch.sum(torch.eq(targets:byte(),pred:ge(0.5)))/pred:size(1) )
   --for i= 1,1000 do
   --image.save('images/image'.. i.. '.png', d_inp[i]:view(28,28))
   --end
   image.save('images/image'.. j.. '.png', d_inp[1]:view(154,154))
end

--test the network



--t=mlp:forward(test_inputs)
--mx,ind=t:max(2)
--print(torch.sum(torch.eq(ind:view(10000),test_targets:long()))/10000)
