if ... then
  module(..., package.seeall)
end


function shuffle(x)
   s = x:storage()
   return torch.randperm(#s)
      :apply(function(i) return s[i] end)
      :reshape(x:size())
end
torch.Tensor.shuffle = shuffle

function test_shuffle()
   local a = torch.randn(2,3)
   local b = a:shuffle()
   local as = a:reshape(#a:storage()):sort()
   local bs = b:reshape(#b:storage()):sort()
   assert(as:eq(bs):prod() == 1)
end

if not ... then -- main
   test_shuffle()
end
