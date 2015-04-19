require 'nngraph';
require 'nn';

-- set seed
torch.manualSeed(123)

-- write an nngraph gModule that computes z = x1 + x2 * linear(x3) 
x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Linear(5,5)()

-- multiply elementwise x2 and linear(x3)
m23 = nn.CMulTable()({x2,x3})
-- add elementwise x1 and the result of m23
m123 = nn.CAddTable()({x1,m23})
--put it all together
mod = nn.gModule({x1,x2,x3},{m123})

-- define the tensors to use
t1 = torch.Tensor{1,2,3,4,5}
t2 = torch.Tensor{2,4,6,8,10}
t3 = torch.Tensor{1,3,5,7,9}

-- run the tensors through the gModule
outg = mod:forward({t1,t2,t3})

-- generate the same output via regular feed forward modules
test = nn.Linear(5,5)
out = torch.add(t1,torch.cmul(t2,test:forward(t3)))

--compare outputs
print('Output via gModule')
print(outg)
print('Output via ff module')
print(out)