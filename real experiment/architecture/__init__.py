import torch
from .MFMLP import MFMLP

def model_generator(method,pretrained_model_path):
    if method in['mfmlp_2stg']:
        model=MFMLP(stage=2,dim=28,depth=2,num_blocks=[1,1,1]).cuda()
    elif method in['mfmlp_5stg']:
        model=MFMLP(stage=5,dim=28,depth=2,num_blocks=[1,1,1]).cuda()
    elif method in['mfmlp_11stg']:
        model=MFMLP(stage=11,dim=28,depth=2,num_blocks=[1,1,1]).cuda()

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        # checkpoint = torch.load(pretrained_model_path)
        checkpoint = torch.load(pretrained_model_path)['model_dict']
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
        #                       strict=True)
        model_dict=model.state_dict()
        model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint.items() if k.replace('module.','') in model_dict },
                              strict=True)

    return model



