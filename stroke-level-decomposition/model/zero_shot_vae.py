import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import shutil
import datetime
import pickle as pkl
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')  # 将上层目录加入环境变量
from data.lmdbReader import lmdbDataset, resizeNormalize
writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

class FewShotDataset(Dataset):

    def __init__(self, pair):
        self.samples = pair

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Encoder(nn.Module):

    def __init__(self, latent_space):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(32*32,1024)
        self.linear2 = nn.Linear(1024,1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512,latent_space)
        self.relu = nn.ReLU()

    def forward(self, sequence):
        batch = sequence.shape[0]
        x = self.linear1(sequence.view(batch,-1))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

class Decoder(nn.Module):

    def __init__(self,latent_space):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_space,2048)
        self.linear2 = nn.Linear(2048,1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence):
        batch = sequence.shape[0]

        x = self.linear1(sequence)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x).view(batch,1,32,32)
        # x = self.sigmoid(self.linear2(self.relu(self.linear1(sequence)))).view(batch, 1, 28, 28)
        return x


class VAE(nn.Module):

    def __init__(self,latent_space):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_space)
        self.decoder = Decoder(latent_space)
        self.project2mu = nn.Linear(latent_space, latent_space)
        self.project2logsigma = nn.Linear(latent_space, latent_space)

    def sample_z(self, x):
        mu = self.project2mu(x)
        log_sigma = self.project2logsigma(x)
        sigma = torch.exp(log_sigma)

        self.mu = mu
        self.sigma = sigma

        sample_from_gauss = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        latent = mu + sigma * sample_from_gauss
        return latent, mu

    def forward(self, sequence):
        x = self.encoder(sequence)
        latent, mu = self.sample_z(x)
        image_sequence = self.decoder(latent)
        return image_sequence, mu


'''
这个地方改一下
'''

def get_dataset(root):
    f = open(root, 'rb')
    dataset = pkl.load(f)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=8,
    )
    return dataset, dataloader


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

config = {
    'exp_name' : 'EMNIST实验',
    'lr' : 1e-3,
    'epoch' : 5000,
    'latent_space' : 400,
}

if __name__ == '__main__':
    batch_size = 32
    # clean_dir(config['exp_name'])
    _, dataloader = get_dataset('/home/chenjingye/IJCAI2021/brain_storm/strokelet/data/zero_shot_test_1000/zeroshot_ic13_train_1000.pkl')
    vae = VAE(config['latent_space'])

    vae = vae.cuda()
    # vae = nn.DataParallel(vae)

    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(vae.parameters(), lr=config['lr'])

    for epoch in range(0, config['epoch']):

        # torch.save(vae.state_dict(), '/home/chenjingye/CVPR2021/pytorch-vae/checkpoint/{0}.pth'.format(epoch))

        # 可视化
        # 从正态分布采样得到8个数值
        # latent = torch.from_numpy(np.random.normal(0, 1, size=(1, config['latent_space']))).float().view(1, -1).cuda()
        # generate_image = vae.decoder(latent).view(28, 28, 1)
        #
        # cv2.imwrite('./image/{0}/{1}.jpg'.format(config['exp_name'], epoch), generate_image.detach().cpu().numpy() * 256)

        for iter, data in enumerate(dataloader):
            images, classes = data
            images = torch.Tensor(images).cuda()
            # 转灰度图
            R = images[:,0,:,:]
            G = images[:, 0, :, :]
            B = images[:, 0, :, :]
            images = 0.299 * R + 0.587 * G + 0.114 * B

            batch = images.shape[0]

            optimizer.zero_grad()
            decode_image = vae(images)
            ll = latent_loss(vae.mu, vae.sigma)
            mse = criterion(decode_image, images)
            loss = ll + mse

            loss.backward()
            optimizer.step()

            # 与tensorboard的交互
            if iter % 1000 == 0:
                grid1 = images.view(batch, 1, 1, 32, 32)
                grid2 = decode_image.cuda().view(batch, 1, 1, 32, 32)
                combine = torch.cat([grid1, grid2], 1)
                output = combine.view(batch*2, 1, 32, 32)
                # decode_image = decode_image.view(batch,1,28,28)
                grid = torchvision.utils.make_grid(output,padding=2)
                writer.add_image('images', grid, 0)

                writer.add_scalar('latent_loss', ll, (epoch - 1) * len(dataloader) + iter)
                writer.add_scalar('mse_loss', mse, (epoch - 1) * len(dataloader) + iter)

            l = loss.item()

            if iter % 100 == 0:
                print("Epoch {} | Iter {} | LatentLoss {} | MSELoss {}".format(epoch, iter, ll, mse))
