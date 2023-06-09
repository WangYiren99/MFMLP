from architecture import *
from utils import *
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option import opt
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)
# CAVE= np.zeros((((512,512,28,30))))
# KAIST=np.zeros((((2704,3376,28,30))))

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.out_train = os.path.join(opt.out_train, date_time)
if not os.path.exists(opt.out_train):
    os.makedirs(opt.out_train)

# model

model = model_generator(opt.method).cuda()


# optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)
criterion = nn.L1Loss()
mse = torch.nn.MSELoss().cuda()

# start_epoch=0
if opt.resume:
    checkpoint=torch.load(opt.pretrained_model_path)
    dict=checkpoint['model_dict']
    # for key in list(dict.keys()):
    #     if key in ['module.denoiser.norm.weight','module.denoiser.norm.bias']:
    #         del dict[key]
    model.load_state_dict(dict) 
    optimizer.load_state_dict(checkpoint['optimizer_dict'])  
    start_epoch = checkpoint['epoch']  
    scheduler.load_state_dict(checkpoint['scheduler'])

if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## pipline of training
    for epoch in range(1, opt.max_epoch):
        model.train()
        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    
        epoch_loss = 0

        start_time = time.time()
        for i, (input, label, Mask, Phi, Phi_s) in tqdm(enumerate(loader_train),total=len(loader_train),leave=True):
            input, label, Phi, Phi_s = Variable(input), Variable(label), Variable(Phi), Variable(Phi_s)
            input, label, Phi, Phi_s = input.cuda(), label.cuda(), Phi.cuda(), Phi_s.cuda()

            input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask)
            out = model(input, input_mask)
            loss = torch.sqrt((out, label))
            
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if i % (1000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))
        # if epoch%5 == 0:
        #     CheckPoint(model, epoch, os.path.join(opt.out_train 'model_%03d.pth' % (epoch + 1)),optimizer,scheduler)
        torch.save(model, os.path.join(opt.out_train, 'model_%03d.pth' % (epoch + 1)))
