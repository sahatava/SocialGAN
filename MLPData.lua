require 'image'

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

train_inputs = torch.FloatTensor(10000,1,130,130)
for i = 1,10000 do
train_inputs[i] = image.load('realData/data'.. i ..'.png')
end

--set data for a special number
--num = 4
--s =torch.sum(torch.eq(train_targets,num))


--s = train:size()
s = 10000

--train_inputs = torch.FloatTensor(s,1,28,28)
--local count = 1
--for i=1,50000 do
--   if train_targets[i] == num then
--     train_inputs[count] = TrainNew[i]
--      count = count +1 
--   end
--end
	
--define a network architecture 
require 'nn'
d_mlp = nn.Sequential()
d_mlp:add(nn.View(130*130))
d_mlp:add(nn.Linear(130*130,1000))
d_mlp:add(nn.ReLU())
--d_mlp:add(nn.Dropout(0.4))
--d_mlp:add(nn.Linear(200,200))
--d_mlp:add(nn.ReLU())
--d_mlp:add(nn.Dropout(0.4))
d_mlp:add(nn.Linear(1000,1))
d_mlp:add(nn.Sigmoid())


g_mlp = nn.Sequential()
g_mlp:add(nn.View(100))
g_mlp:add(nn.Linear(100,400))
g_mlp:add(nn.ReLU())
--g_mlp:add(nn.Linear(400,400))
--g_mlp:add(nn.ReLU())
g_mlp:add(nn.Linear(400,130*130))
g_mlp:add(nn.Sigmoid())



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

lr=0.0001
for j = 1,1000 do
   --lr=lr*0.99
   --create dataset
bt=100
   g_inputs=torch.Tensor(s*2,100):normal(0,1)
   inputs=torch.cat(train_inputs,g_mlp:forward(g_inputs):view(g_inputs:size(1),1,130,130),1)
   targets=torch.cat(torch.ones(s),torch.zeros(g_inputs:size(1)))
   --shuffling the data
   ShuffleIndex = torch.randperm(inputs:size(1)):long()
   inputs = inputs:index(1,ShuffleIndex)
   targets = targets:index(1,ShuffleIndex)
   g_targets=torch.ones(g_inputs:size(1))
   d_mlp:training()   
   for i=1,inputs:size(1)/bt do
	d_mlp:zeroGradParameters()	
	x=inputs[{{(i-1)*bt+1,i*bt},{1},{},{}}]
	y=targets[{{(i-1)*bt+1,i*bt}}]
	criterion = nn.BCECriterion()
	pred = d_mlp:forward(x)
	out = criterion:forward(pred, y)
	err = criterion:backward(pred, y)
	d_mlp:backward(x, err)
	d_mlp:updateParameters(lr)
	
   end
pred = d_mlp:forward(inputs)
print(torch.sum(torch.eq(targets:byte(),pred:ge(0.5)))/pred:size(1) )

bt=100	
lr=0.01
   for i=1,g_inputs:size(1)/bt do 
	g_mlp:zeroGradParameters()
	x=g_inputs[{{(i-1)*bt+1,i*bt}}]
	y=g_targets[{{(i-1)*bt+1,i*bt}}]
	criterion = nn.BCECriterion()
	d_inp=g_mlp:forward(x)
	pred = d_mlp:forward(d_inp)
	out = criterion:forward(pred, y)
	err = criterion:backward(pred, y)
	grads=d_mlp:backward(d_inp, err)
	g_mlp:backward(x, grads)
	g_mlp:updateParameters(lr)
   end
   d_mlp:evaluate()
inputs=torch.cat(train_inputs,g_mlp:forward(g_inputs):view(g_inputs:size(1),1,130,130),1)
pred = d_mlp:forward(inputs)
print(torch.sum(torch.eq(targets:byte(),pred:ge(0.5)))/pred:size(1) )
   --for i= 1,1000 do
   --image.save('images/image'.. i.. '.png', d_inp[i]:view(28,28))
   --end
   image.save('images/image'.. j.. '.png', d_inp[1]:view(130,130))
end

--test the network



--t=mlp:forward(test_inputs)
--mx,ind=t:max(2)
--print(torch.sum(torch.eq(ind:view(10000),test_targets:long()))/10000)
