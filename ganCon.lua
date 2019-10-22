require 'image'
require 'nn'
require 'nngraph'


torch.setdefaulttensortype('torch.FloatTensor')
-- load the MNIST dataset
local dl = require 'dataload'

train, valid, test = dl.loadMNIST()
indices = torch.LongTensor(train:size())
for i=1,train:size() do
   indices[i]=i
end
train_inputs, train_targets = train:index(indices)
 

test_inputs, test_targets = test:index(indices[{{1,10000}}])


valid_inputs, valid_targets = valid:index(indices[{{1,10000}}])

--set data for a special number
--num = 4
--s =torch.sum(torch.eq(train_targets,num))
s = train:size()
--train_inputs = torch.FloatTensor(s,1,28,28)
--local count = 1
--for i=1,50000 do
--   if train_targets[i] == num then
--     train_inputs[count] = TrainNew[i]
--      count = count +1 
--   end
--end
	
--define a network architecture 

model_D = nn.Sequential()
model_D:add(nn.SpatialConvolution(1, 32, 5, 5, 1, 1, 4, 4))
model_D:add(nn.SpatialMaxPooling(2,2))
model_D:add(nn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(nn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
model_D:add(nn.SpatialMaxPooling(2,2))
model_D:add(nn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(nn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
model_D:add(nn.ReLU(true))
model_D:add(nn.SpatialMaxPooling(2,2))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(nn.Reshape(4*4*96))
model_D:add(nn.Linear(4*4*96, 1024))
model_D:add(nn.ReLU(true))
model_D:add(nn.Dropout())
model_D:add(nn.Linear(1024,1))
model_D:add(nn.Sigmoid())

 x_input = nn.Identity()()
lg = nn.Linear(512, 128*8*8)(x_input)
lg = nn.Reshape(128, 8, 8)(lg)
lg = nn.ReLU(true)(lg)
lg = nn.SpatialUpSamplingNearest(2)(lg)
lg = nn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2)(lg)
lg = nn.SpatialBatchNormalization(256)(lg)
lg = nn.ReLU(true)(lg)
lg = nn.SpatialUpSamplingNearest(2)(lg)
--lg = cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2)(lg)
--lg = nn.SpatialBatchNormalization(256)(lg)
--lg = cudnn.ReLU(true)(lg)
--lg = nn.SpatialUpSamplingNearest(2)(lg)
lg = nn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2)(lg)
lg = nn.SpatialBatchNormalization(128)(lg)
lg = nn.ReLU(true)(lg)
lg = nn.SpatialConvolution(128, 3, 3, 3, 1, 1, 1, 1)(lg)
model_G = nn.gModule({x_input}, {lg})








--train the network


--print(nn.SoftMax():forward(mlp:forward(train_inputs[1])))

lr=0.01
for j = 1,100 do
   --lr=lr*0.99
   --create dataset
bt=10
   g_inputs=torch.Tensor(s,512):normal(0,1)
   inputs=torch.cat(train_inputs,model_G:forward(g_inputs):view(s,1,28,28),1)
   targets=torch.cat(torch.ones(s),torch.zeros(s))
   g_targets=torch.ones(s)
   model_D:training()   
   for i=1,inputs:size(1)/bt do
	model_D:zeroGradParameters()	
	x=inputs[{{(i-1)*bt+1,i*bt},{1},{},{}}]
	y=targets[{{(i-1)*bt+1,i*bt}}]
	criterion = nn.BCECriterion()
	pred = model_D:forward(x)
	out = criterion:forward(pred, y)
	err = criterion:backward(pred, y)
	model_D:backward(x, err)
	model_D:updateParameters(lr)
	
   end

bt=10	
   for i=1,g_inputs:size(1)/bt do 
	model_G:zeroGradParameters()
	x=g_inputs[{{(i-1)*bt+1,i*bt}}]
	y=g_targets[{{(i-1)*bt+1,i*bt}}]
	criterion = nn.BCECriterion()
	d_inp=model_G:forward(x)
	pred = model_D:forward(d_inp)
	out = criterion:forward(pred, y)
	err = criterion:backward(pred, y)
	grads=model_D:backward(d_inp, err)
	model_G:backward(x, grads)
	model_G:updateParameters(lr*0.1)
   end
   model_D:evaluate()
  
   image.save('image'.. j.. '.png', d_inp[1]:view(28,28))
end

--test the network



--t=mlp:forward(test_inputs)
--mx,ind=t:max(2)
--print(torch.sum(torch.eq(ind:view(10000),test_targets:long()))/10000)
