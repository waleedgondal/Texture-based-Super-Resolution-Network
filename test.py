import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from scipy.misc import imread, imresize
import torchvision

# Test Settings
parser = argparse.ArgumentParser(description='PyTorch Texture Based Single Image Super-Resolution')
parser.add_argument('--path_model', type=str, default='./resources/pretrained/tsrn_segment_8x.pth', help='path for pretrained models.')
parser.add_argument('--path_data', type=str, default='./resources/images', help='path of test images.')
parser.add_argument('--path_output', type=str, default='./output/images', help='output path for images generated.')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor. Default=4")
parser.add_argument('--cuda', type=int, default=1, help='Try to use cuda? Default=1')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use. Default=4')
opt = parser.parse_args()

if opt.upscale_factor == 4: 
    from model import Model_4x as Model
elif opt.upscale_factor == 8: 
    from model import Model_8x as Model

# Make sure that the upscale factor matches the upsampling model being used
#model_upscalefactor = ((str(opt.path_model)).split('_')[-1].split('x')[0]).split(' ')[-1]
#assert (opt.upscale_factor == int(model_upscalefactor)),"Given upscaling factor doesn't match the given model's upsampling!" 

if opt.cuda:
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
        print('===> Warning: failed to load CUDA, running on CPU!')

def read_Images(root):
    images_list = []
    for file in os.listdir(root):
        images_list.append(os.path.join(root, file))
    return images_list

def to_variable(x):
    """Convert tensor to variable."""
    if cuda:
        x = x.cuda()
    return Variable(x)

print ('======> Images are being read')
images =read_Images(opt.path_data)
output_path = os.path.join(opt.path_output,str(opt.upscale_factor)+'x/')
if not os.path.exists(output_path):
    os.makedirs(output_path)
print (' Outputs at: ',opt.path_output)

print ('======> Super Resolution Model is being Loaded')
sr_net = Model(img_channels=3)
model_dict = sr_net.state_dict()
sr_net.load_state_dict(torch.load(opt.path_model, map_location=lambda storage, loc: storage))
if cuda:
    sr_net = sr_net.cuda()
sr_net.eval()

upsample = opt.upscale_factor
img_transform = transforms.Compose([transforms.ToTensor()])
for i, image in enumerate(images):
    name = image.split('/')[-1]
    img = imread(image)
    #img = imresize(img, (img.shape[0]//upsample,img.shape[1]//upsample), interp='bicubic')
    bicubic_img = imresize(img, (img.shape[0]*upsample,img.shape[1]*upsample), interp='bicubic')
    
    img_t = img_transform(img)
    img_t = to_variable(img_t.view(1, img_t.size(0), img_t.size(1), img_t.size(2)))
    bicubic_t = img_transform(bicubic_img)
    bicubic_t = to_variable(bicubic_t.view(1, bicubic_t.size(0), bicubic_t.size(1), bicubic_t.size(2)))
    print ('======> Super Resolution Predictions are being made')
    data_out = sr_net(img_t)
    data_out = data_out + bicubic_t
    torchvision.utils.save_image((data_out.data).clamp(0, 1),os.path.join(output_path,name))