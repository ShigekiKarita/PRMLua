if ... then
   module(..., package.seeall)
end


do -- RVM namespace
   local RVM = torch.class('RVM')

   function RVM:__init(kernel)
      require "kernels"
      self.kernel = kernel or kernels.gaussian_kernel
   end

   function RVM:fit(input_list, target_list, tol, limit)
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
         local weight_score = torch.ones(n_data + 1) - torch.cmul(self.weight_precision, self.weight_covariance:diag())
         local new_tp = (n_data - weight_score:sum()) / (target_list - design_matrix * self.weight_mean):norm()^2
         local new_wp = weight_score:cdiv(torch.pow(self.weight_mean, 2)):clamp(-math.huge, self.limit)
         diff = torch.norm(new_wp - self.weight_precision)^2 + (new_tp - self.target_precision)^2
         self.weight_precision, self.target_precision = new_wp, new_tp
      end

      return self
   end

   function RVM:mean(input)
      function k(x) return self.kernel(input, x) end
      local kx = torch.ones(1):cat(self.xs:clone():apply(k))
      return self.weight_mean * kx
   end

   function RVM:variance(input)
      function k(x) return self.kernel(input, x) end
      local kx = torch.ones(1):cat(self.xs:clone():apply(k))
      return kx * (self.weight_covariance * kx) + 1.0 / self.target_precision
   end

   function RVM:predict(input)
      if torch.type(input) == "number" then
         return torch.randn(1)[1] * self:variance(input) + self:mean(input)
      else
         return input:clone():apply(function(x) return self:predict(x) end)
      end
   end
end -- RVM namespace



if not ... then -- main
   require "gnuplot"
   require "mytorch"
   require "kernels"

   local window_id = 0
   function test_rvm(kernel)
      torch.manualSeed(0)
      local x = torch.range(-0.0, 10.0, 0.1)
      x = x + torch.randn(#x) * 2.0
      local y = torch.sin(x) + torch.randn(#x) * 0.1
      local r = RVM(kernel)
      r:fit(x, y)
      local test_x = torch.range(-10.0, 20.0, 0.1)
      local mean_y = test_x:clone():apply(function(x) return r:mean(x) end)
      local var_y = test_x:clone():apply(function(x) return r:variance(x) end) * 10.0
      local test_y = r:predict(test_x)

      window_id = window_id + 1
      gnuplot.figure(window_id)
      gnuplot.plot(
         {"sample", x, y, "+"},
         {"mean", test_x, mean_y, "-"},
         {"var+", test_x, mean_y + var_y, "-"},
         {"var-", test_x, mean_y - var_y, "-"}
      )
   end

   test_rvm(kernels.gaussian_kernel)
   gnuplot.title("gaussian-kernel RVM")

   test_rvm(kernels.sigmoid_kernel)
   gnuplot.title("sigmoid-kernel RVM")
end
