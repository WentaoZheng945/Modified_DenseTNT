import torch
from torch import nn

class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        print('x.shape:', x.shape)
        u = x.mean(-1, keepdim=True)
        print('u.shape:', u.shape)
        print(u)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


if __name__ == '__main__':
    # ln = LayerNorm(hidden_size=10)
    # random_input = torch.randn(5, 10)
    # output = ln(random_input)
    # tensor_33 = torch.tensor([[1, 2, 3],
    #                           [4, 5, 6],
    #                           [7, 8, 9]], dtype=torch.float)
    # print(random_input)
    # print(output)
    # ln2 = LayerNorm(3)
    # output2 = ln2(tensor_33)
    # print(output2)
    high_tensor = torch.arange(1, 55, 1.0)
    high_tensor.resize_(2, 3, 9)
    print(high_tensor.dtype)
    print(high_tensor)
    new_ln = LayerNorm(9)
    new_output = new_ln(high_tensor)
    print(high_tensor[0])
    print(new_output[0])
    # high_tensor = torch.randn()