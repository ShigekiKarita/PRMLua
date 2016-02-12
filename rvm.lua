if ... then
  module(..., package.seeall)
end


local RVM = {}

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


function RVM.new(kernel)
   local self = {}
   self.kernel = kernel or sigmoid_kernel
   return setmetatable(self, {__index = RVM})
end

function RVM.fit(self, input_list, target_list, tol, limit)
   local n_data = input_list:size(1)
   self.xs = input_list
   self.weight_precision = torch.ones(n_data + 1)
   self.target_precision = 1.0
   self.limit = limit or 1e2
   local tol = tol or 1e-6

   local design_matrix = torch.ones(n_data, n_data + 1)
   for i = 1, n_data do
      x1 = input_list:sub(i,i)
      for j = 1, n_data do
         x2 = input_list:sub(j,j)
         design_matrix[i][j+1] = self.kernel(x1, x2)
      end
   end

   local diff = math.huge
   while diff >= tol do
      -- Expect a posterior: p (w|target_list, input_list, w_prec, t_prec) = Normal(w|w_mean, w_var)
      local l = self.weight_precision:diag()
      local r = design_matrix:t() * design_matrix * self.target_precision
      self.weight_covariance = torch.inverse(l + r)
      self.weight_mean = self.weight_covariance * (design_matrix:t() *  target_list) * self.target_precision

      -- Maximize an evidence: p(target | input_list, w_mean, w_var) by steepest descent: d (log evidence) / dw = 0
      local weight_score = torch.ones(n_data + 1) -  torch.cmul(self.weight_precision, self.weight_covariance:diag())
      local new_tp = (n_data - weight_score:sum()) / (target_list - design_matrix * self.weight_mean):norm()^2
      local new_wp = weight_score:cdiv(torch.pow(self.weight_mean, 2)):clamp(-math.huge, self.limit)
      diff = torch.norm(new_wp - self.weight_precision)^2 + (new_tp - self.target_precision)^2
      self.weight_precision, self.target_precision = new_wp, new_tp
   end

   return self
end

function RVM.mean(self, input)
   function k(x) return self.kernel(input, x) end
   local kx = torch.ones(1):cat(self.xs:clone():apply(k))
   return self.weight_mean * kx
end

function RVM.variance(self, input)
   function k(x) return self.kernel(input, x) end
   local kx = torch.ones(1):cat(self.xs:clone():apply(k))
   return kx * (self.weight_covariance * kx) + 1.0 / self.target_precision
end

function RVM.predict(self, input)
   if torch.type(input) == "number" then
      return torch.randn(1)[1] * self:variance(input) + self:mean(input)
   else
      return input:clone():apply(function(x) return self:predict(x) end)
   end
end


-- Example
function test_rvm(kernel)
   require "gnuplot"
   require "mytorch"

   torch.manualSeed(0)
   x = torch.range(-0.0, 10.0, 0.1)
   x = x + torch.randn(#x) * 2.0
   y = torch.sin(x) + torch.randn(#x) * 0.1

   r = RVM.new(kernel)
   r:fit(x, y)
   test_x = torch.range(-10.0, 20.0, 0.1)
   mean_y = test_x:clone():apply(function(x) return r:mean(x) end)
   var_y = test_x:clone():apply(function(x) return r:variance(x) end) * 10.0
   test_y = r:predict(test_x)

   -- gnuplot.pngfigure('plot.png')
   gnuplot.plot(
      {"sample", x, y, "+"},
      {"mean", test_x, mean_y, "-"},
      {"var+", test_x, mean_y + var_y, "-"},
      {"var-", test_x, mean_y - var_y, "-"}
   )
   -- gnuplot.plotflush()
end



if not ... then -- main
   require "gnuplot"
   gnuplot.figure(1)
   gnuplot.title("gaussian-kernel RVM")
   test_rvm(gaussian_kernel)

   gnuplot.figure(2)
   gnuplot.title("sigmoid-kernel RVM")
   test_rvm(sigmoid_kernel)
end
