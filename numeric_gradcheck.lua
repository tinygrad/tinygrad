require 'torch';
require 'math'
require 'io'
require 'cutorch';
require 'cunn';


opt = 'DOUBLE'  -- FLOAT / DOUBLE / CUDA

-----------------------------------------------------------
--function to customize memory type
--

local function customize(memory)

    if(opt == 'FLOAT') then
        return memory:float()
    elseif(opt == 'CUDA') then
        return memory:cuda()
    else
        --do nothing and just return
        -- by default, it is double
        return memory
    end
end

--------------------------------------------------------------
-- INPUT SETTINGS
-- 

tensor = torch.DoubleTensor(2, 3, 4);
tensor = customize(tensor:rand(2, 3, 4) * 10)

data = {
    inputs = tensor,
    targets = customize(torch.DoubleTensor({1, 0}))
}


print(tensor)

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------
  local n_classes = 2

  -- MODEL:
  local model = nn.Sequential()
  model:add(nn.SpatialConvolutionMM(2, 5, 3, 3, 1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(5, 2, 3, 3, 1, 1, 1, 1))

  model:add(nn.Reshape(2 * 3  * 4))
  model:add(nn.Linear(2 * 3  * 4, n_classes + 10))
  model:add(nn.Linear(n_classes + 10, n_classes))
  model:add(nn.LogSoftMax())
  
  customize(model)
  
  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.ClassNLLCriterion()
  customize(criterion)
 
 if model then
    parameters,gradParameters = model:getParameters()
 end

--------------------------------------------------------------------------------
-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)
  
  -- compute numeric approximations to gradient
  local eps = 1e-7
  local grad_est = customize(torch.DoubleTensor(grad:size()))
  grad_est:zero()

  for i = 1, grad:size(1) do
    -- do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    
    --create a temporary tensor for X to hold the 'eps' in the appropriate position 
    tempX = customize(torch.DoubleTensor(grad:size(1)))
    tempX:zero()
    tempX[i] = eps
    
    -- calculate delta parameters for gradient calculation
    x_plus_eps = x + tempX
    x_minus_eps = x - tempX
    
    -- by using delta set of parameters, estimate the gradient for particular parameter 
    gradient = (f(x_plus_eps) - f(x_minus_eps)) / (eps * 2);
    grad_est[i] = gradient
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / (2 * torch.norm(grad + grad_est))
  return diff, grad, grad_est
end

---------------------------------------------------------------------------
-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  return criterion:forward(model:forward(data.inputs), data.targets)
end

--------------------------------------------------------------------------
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  
  gradParameters:zero()

  local outputs = model:forward(data.inputs)
  criterion:forward(outputs, data.targets)
  model:backward(data.inputs, criterion:backward(outputs, data.targets))

  return gradParameters
end

--------------------------------------------------------------------
local writeFile = function(filename, data)
    -- Opens a file in append mode
    file = io.open(filename, "w")

    -- sets the default output file as test.lua
    io.output(file)

    for index = 1, data:size(1) do
        number = data[index]
        -- appends a word test to the last line of the file
        
        io.write(string.format('%.4f\n', number))
    end
    
    -- closes the open file
    io.close(file) 
end

print 'checking gradient ...'

-- call the checkgrad function to get the actual and estimate of the gradient 
local diff, grad, est = checkgrad(f, g, parameters)

-- print the actual gradient from the predefined criterion
print('actual gradient : \n')
writeFile('actual_grad.txt', grad)

-- print the estimated gradient from the approximation method
print('estimated gradient : \n')
writeFile('estimate_grad.txt', est)

--variables to find cosine similarity 
nominator = torch.sum(torch.cmul(grad, est))
denominator =  ((torch.norm(grad)) * torch.norm(est))

local cosineSimilarity = nominator / denominator

--print the status to console
print('symmetric relative error : ' .. diff .. ' --> cosine similarity : ' .. cosineSimilarity..'\n\n')
