import torch
print(torch.backends.cudnn.version())
# 正确返回8200

from torch.backends import cudnn
print(cudnn.is_available())
# 若正常返回True

a = torch.tensor(1.)
print(cudnn.is_acceptable(a.cuda()))
# 若正常返回True