import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction")
parser.add_argument('--template', default='mfmlp',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../datasets/simulation_dataset/', help='dataset directory')

# Saving specifications
parser.add_argument('--out_train', type=str, default='./exp_train/mfmlp_2stg/', help='saving path for training')
parser.add_argument('--out_test', type=str, default='./exp_test/mfmlp_2stg/', help='saving path for testing')

# Model specifications
parser.add_argument('--method', type=str, default='mfmlp_2stg', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi_PhiPhiT',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument('--resume',type=bool,default=False,help='Continue training from the last saved model') 

opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"{opt.data_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/simu_mask/"
opt.test_path = f"{opt.data_root}/KAIST_10/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False