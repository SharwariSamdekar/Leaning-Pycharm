import torch

# ====================================================================================== #
#                    Tensor Math and Comparison Operations                               #
# ====================================================================================== #

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)

z2 = torch.add(x,y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x,y)
print(z)

# Inplace Operations
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparisons

z = x  > 0
z = x < 0
print(z)

# Matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# Matrix exponentiation
matrix_exp =torch.rand((5,5))
print(matrix_exp.matrix_power(3))

# element wise mult
z = x * y
print(z)

# Dot product
z = torch.dot(x,y)
print(z)

# Batched matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1, tensor2) # batch, m, p

# Example of broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2
print(z)

# other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) # can also be used as x.max(dim=0)
values, indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y)
print(z)

sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10) # set all elements of x which are less than zero to zero and max to ten

# Relu is special case of clamp. We can say torch.clamp(x, min=0) then it will work like relu

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) # returns true if any one true (like or function)
z = torch.all(x) # returns true if all are true

