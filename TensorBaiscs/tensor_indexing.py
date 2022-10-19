import torch

# ====================================================================================== #
#                                   Tensor Indexing                                      #
# ====================================================================================== #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x[0,:]

print(x[:,0].shape)

print(x[2, 0:10]) # 0:10 -----> [0,1,2......9]

x[0,0] = 100

# Fancy Indexing
x = torch.arange(10)
indices = [2,5,8]
print(x)
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x)
print(x[rows,cols]) # It will first print the 2nd row and 5th column and then 0th row 0th column elements

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0]) # if x % 2  == 0 then print

# Useful operations
print(torch.where(x > 5, x, x*2)) # Like a ternary operator if x > 5 is false then x*2 else x
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())
print(x.numel()) # count the number of elements in x