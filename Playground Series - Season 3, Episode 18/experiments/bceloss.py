import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()

# input = torch.randn(3, 3, requires_grad=True)
input = torch.tensor([
        [-0.6684,  0.4074,  0.4228],
        [-1.1824,  1.0974, -0.2524],
        [-0.0323, -0.2636,  0.6600]
    ], requires_grad=True)

m_input = m(input)
# tensor([[0.3389, 0.6005, 0.6042],
#         [0.2346, 0.7498, 0.4372],
#         [0.4919, 0.4345, 0.6593]], grad_fn=<SigmoidBackward0>)

# target = torch.empty(3, 3).random_(2)
target = torch.tensor([
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 0.]
    ])

output = loss(m_input, target)
# tensor(0.7227, grad_fn=<BinaryCrossEntropyBackward0>)
output.backward()

result = -1.0/9*(
    torch.log(torch.Tensor([1-0.3389]))+
    torch.log(torch.Tensor([1-0.6005]))+
    torch.log(torch.Tensor([0.6042]))+
    torch.log(torch.Tensor([0.2346]))+
    torch.log(torch.Tensor([0.7498]))+
    torch.log(torch.Tensor([1- 0.4372]))+
    torch.log(torch.Tensor([0.4919]))+
    torch.log(torch.Tensor([1- 0.4345]))+
    torch.log(torch.Tensor([1- 0.6593]))
)
# tensor([0.7227])