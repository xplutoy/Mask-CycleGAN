import os

import torch as T
import torchvision as tv
from torch.autograd import Variable

from coco_sub import apple2orange_loader
from networks import ResnetGenerator

epoch = 100

# data
save_dir = './valid_results/'
os.makedirs(save_dir, exist_ok=True)

# network
input_nc = 3
output_nc = 3
ngf = 64
ndf = 64

netG_A = ResnetGenerator(input_nc, output_nc, ngf).cuda()
netG_B = ResnetGenerator(input_nc, output_nc, ngf).cuda()

netG_A.eval()
netG_B.eval()

if epoch >= 1:
    checkpoint = T.load(save_dir + 'ckpt_{}.ptz'.format(epoch))
    netG_A.load_state_dict(checkpoint['G_A'])
    netG_B.load_state_dict(checkpoint['G_B'])

batch = 0
for A, B in apple2orange_loader:
    batch += 1

    a_real = Variable(A, volatile=True).cuda()
    b_fake = netG_A(a_real)
    a_rec = netG_B(b_fake)

    b_real = Variable(B, volatile=True).cuda()
    a_fake = netG_B(b_real)
    b_rec = netG_A(a_fake)

    tv.utils.save_image(T.cat([
        a_real.data * 0.5 + 0.5,
        b_fake.data * 0.5 + 0.5,
        a_rec.data * 0.5 + 0.5,
        b_real.data * 0.5 + 0.5,
        a_fake.data * 0.5 + 0.5,
        b_rec.data * 0.5 + 0.5], 0),
        save_dir + 'valid_{}_{}.png'.format(epoch, batch), 3)
