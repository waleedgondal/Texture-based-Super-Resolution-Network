#!/usr/bin/env python3
import argparse
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import VGG
import torchvision
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Texture Based Single Image Super-Resolution')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor. Default=4")
parser.add_argument('--nof_blocks', type=int, default=10, help="number of blocks in generator. Default=10")
parser.add_argument('--feat_size', type=int, default=64, help="number of feature channels. Default=64")
parser.add_argument('--cropsize', type=int, default=256, help="image cropsize. Default=128")
parser.add_argument('--batch_size', type=int, default=32, help='training batch size. Default=32')
parser.add_argument('--epochs', type=int, default=110, help='number of epochs to train for. Default=110')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate. Default=0.0005')
parser.add_argument('--loss_mse', type=float, default=1, help='weight of MSE loss. Default=1')
parser.add_argument('--loss_texture', type=float, default=1, help='weight of GAN loss. Default=1')
parser.add_argument('--texture_layers', nargs='+', default=['8','17','26','35'], help='vgg layers for texture. Default:[]')
parser.add_argument('--resume',default = None, help='Provide an epoch to resume training from. Default=None')
parser.add_argument('--cuda', type=int, default=1, help='Try to use cuda? Default=1')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use. Default=4')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
parser.add_argument('--path_data', type=str, default='/is/cluster/shared/mscoco/train2014/', help='path of train images.')
opt = parser.parse_args()

print(opt)
print('===> Loading datasets')
from dataset import SISR_Dataset
train_set = SISR_Dataset(path=opt.path_data, cropsize= opt.cropsize,upscale = opt.upscale_factor, bicubic = True)
print('===> Found %d training images.' % len(train_set))
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                  batch_size=opt.batch_size, shuffle=True)

if opt.upscale_factor == 4: 
    from model import Model_4x as Model
elif opt.upscale_factor == 8: 
    from model import Model_8x as Model
    
torch.manual_seed(opt.seed)
if opt.cuda:
    if torch.cuda.is_available():
        cuda = True
        torch.cuda.manual_seed(opt.seed)
    else:
        cuda = False
        print('===> Warning: failed to load CUDA, running on CPU!')

print('===> Building model')
if opt.resume:
    print('Using Pre-trained Model')
    G = torch.load(opt.resume)
    G.train()
else:
    G = Model(img_channels=3, nof_blocks=opt.nof_blocks, feat_size=opt.feat_size)
if cuda:
    G = G.cuda()

print('===> Building Loss')
def criterion(a, b):
    return torch.mean(torch.abs((a-b)**2).view(-1))
l2_loss = nn.MSELoss().cuda()

if opt.loss_texture:
    vgg_layers = [int(i) for i in opt.texture_layers]
    vgg_texture = VGG(layers=vgg_layers, replace_pooling = False)
    if cuda:
        vgg_texture = vgg_texture.cuda()

print('===> Building Optimizer')
optimizer = optim.Adam(G.parameters(), lr=opt.lr)

def to_variable(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def adjust_learning_rate(optimizer, opt):
    opt.lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

sample_path = './training/'+str(opt.upscale_factor)+'x_global/samples'
saves_path = './training/'+str(opt.upscale_factor)+'x_global/saves'

if not os.path.exists(sample_path):
    os.makedirs(sample_path)
    os.makedirs(saves_path)

print('===> Initializing Training')
def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data_in = to_variable(batch[0])
        data_real = to_variable(batch[1])
        bicubic =  to_variable(batch[2])
        data_fake = G(data_in) + bicubic
        text_loss = []
        if epoch<10:
            text_loss = l2_loss( data_fake,data_real)
        else:
            vgg_fake = vgg_texture.forward(data_fake)
            vgg_real = vgg_texture.forward(data_real)
            gram_fake = [gram_matrix(y) for y in vgg_fake]
            gram_real = [gram_matrix(y) for y in vgg_real]
            for m in range(0, len(vgg_fake)):
                text_loss += [criterion(gram_fake[m], gram_real[m])]
            text_loss = sum(text_loss)

        loss = text_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration%50 == 0:
            print('Epoch [%d], Step[%d/%d], overall_loss: %.8f' 
              %(epoch, iteration, len(training_data_loader), loss.data[0]))
        # Storing samples generated while training
        if iteration in [500,3000]:
            torchvision.utils.save_image((data_fake.data).clamp(0, 1),
                                         os.path.join(sample_path,
                                                      'HR_samples_-%d-%d.png' %(epoch+1, iteration+1)))

def save_checkpoint(epoch):
    G_out_path ='%s/epoch_%s.pth'%(saves_path,str(epoch))
    if not os.path.exists(os.path.dirname(G_out_path)):
        os.makedirs(os.path.dirname(G_out_path))
    torch.save(G, G_out_path)
    print("Checkpoint saved to {}".format(G_out_path))

for epoch in range(1, opt.epochs + 1):
    train(epoch)
    if epoch % 1 == 0:
        save_checkpoint(epoch)
