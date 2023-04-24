import torch
from fvcore.nn import FlopCountAnalysis
from architecture.MFMLP import MFMLP
def my_summary(test_model, H = 256, W = 256,  N = 1):
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, H, W)).cuda()
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')
my_summary(MFMLP(stage=2))