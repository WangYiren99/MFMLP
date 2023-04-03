from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


def torch_psnr_meas(img, ref):  # input [256,310]
    max=torch.max(img)  
    mse = torch.mean((img - ref) ** 2)
    psnr = 10 * torch.log10((max**2)/mse)
    return psnr 
def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    psnr_list, ssim_list = [], []
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)   
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print(f'psnr_mean:{psnr_mean}. ssim_mean:{ssim_mean}. \n')
    print(f"psnr_list:{psnr_list}.\n ssim_list:{ssim_list}")
    model.train()
    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    # model
    model = model_generator(opt.method,'model.pth').cuda()
    pred, truth,psnr_list, ssim_list, psnr_mean, ssim_mean = test(model)
    name = '{}_{:.2f}_{:.3f}'.format(opt.method, psnr_mean, ssim_mean) + '.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})
    scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_list, 'ssim_list': ssim_list})

if __name__ == '__main__':
    main()