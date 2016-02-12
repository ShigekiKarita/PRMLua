if ... then
  module(..., package.seeall)
end


function gaussian_kernel(x1, x2, width)
   local w = width or 1.0
   local d = x1 - x2
   return torch.exp(- torch.pow(torch.sqrt(d * d), 2) / w)
end

function polynominal_kernel(x1, x2, constant, power)
   local c = constant or 1.0
   local p = power or 1.0
   local d = x1 - x2
   return (x1 * x2 + c)^p
end

function sigmoid_kernel(x1, x2, gain)
   local g = gain or 1.0
   return 1.0 / (1.0 + torch.exp(- x1 * x2 * g))
end

function tanh_kernel(x1, x2, gain)
   local g = gain or 1.0
   return torch.tanh(x1 * x2 * g)
end
