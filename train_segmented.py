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
from dataset_segmented import SISR_Dataset_Segment

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Texture Based Single Image Super-Resolution')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor. Default=4")
parser.add_argument('--nof_blocks', type=int, default=10, help="number of blocks in generator. Default=10")
parser.add_argument('--feat_size', type=int, default=64, help="number of feature channels. Default=64")
parser.add_argument('--cropsize', type=int, default=256, help="image cropsize. Default=128")
parser.add_argument('--batch_size', type=int, default=32, help='training batch size. Default=32')
parser.add_argument('--epochs', type=int, default=110, help='number of epochs to train for. Default=110')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate. Default=0.0005')
parser.add_argument('--segmented', type=int, default=1, help='train with segmentation guidance. Default=1')
parser.add_argument('--annFile', type=str, default='/home/wgondal/coco_stuff/stuff_train2017.json', help='Provide MS-COCO Stuff Segmented Annotation File. Default=0')
parser.add_argument('--loss_mse', type=float, default=1, help='weight of MSE loss. Default=1')
parser.add_argument('--loss_texture', type=float, default=1, help='weight of GAN loss. Default=1')
parser.add_argument('--texture_layers', nargs='+', default=['8','17','26','35'], help='vgg layers for texture. Default:[]')
parser.add_argument('--resume',default = None, help='Provide an epoch to resume training from. Default=None')
parser.add_argument('--cuda', type=int, default=1, help='Try to use cuda? Default=1')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use. Default=4')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
parser.add_argument('--path_data', type=str, default='/agbs/cpr/train2017/', help='path of train images.')
opt = parser.parse_args()

if opt.segmented and not opt.annFile or opt.annFile and not opt.segmented:
    parser.error('For semantically guided training, provide path to MS_COCO Annotation File and make segmented flag=1')
print(opt)

print('===> Loading datasets')
train_set = SISR_Dataset_Segment(path=opt.path_data, cropsize= opt.cropsize,upscale = opt.upscale_factor, 
                                     annotationFile = opt.annFile, bicubic = True)

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
criterion = torch.nn.MSELoss()
criterion = criterion.cuda()

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

sample_path = './training/'+str(opt.upscale_factor)+'x_segmented/samples'
saves_path = './training/'+str(opt.upscale_factor)+'x_segmented/saves'

if not os.path.exists(sample_path):
    os.makedirs(sample_path)
    os.makedirs(saves_path)

print('===> Initializing Training')
def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data_in = to_variable(batch[0])                         #[batch, 3, 64, 64]
        data_real = to_variable(batch[1])                       #[batch, 3, 256, 256]
        target_segs = batch[2]                                  #list of 6 with each item [batch, 1,256, 256]
        bicubic_interpolated = to_variable(batch[3])
        data_fake = G(data_in) + bicubic_interpolated
        target_segs = [to_variable(masks) for masks in target_segs]
        text_loss = []
        if epoch<10:
            text_loss = criterion(data_fake,data_real)
        else:
            for mask in target_segs:
                fake_segmented = data_fake * mask
                real_segmented = data_real * mask
                vgg_fake = vgg_texture.forward(fake_segmented)
                vgg_real = vgg_texture.forward(real_segmented)
                gram_fake = [gram_matrix(y) for y in vgg_fake]
                gram_real = [gram_matrix(y) for y in vgg_real]
                for m in range(0, len(vgg_fake)):
                    texture_loss += [criterion(gram_fake[m], gram_real[m])]
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
