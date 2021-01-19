import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

"""
构建vae的模型
"""


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(3, 32)

        self.input = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=8,
                      kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            # nn.MaxPool1d(kernel_size=2, padding=1),
            nn.LeakyReLU()
        )  # (b, 8, 3) ---> b, 24

        # self.fc21 = nn.Linear(32, 8)  # mean
        # self.fc22 = nn.Linear(32, 8)  # var

        # self.fc21 = nn.Linear(24, 8)  # mean
        # self.fc22 = nn.Linear(24, 8)  # var

        self.out = nn.Linear(24, 16)

        # self.fc3 = nn.Linear(8, 32)

        # 将b, 8--->, b, 1, 8
        self.dconv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)  # b, 1, 23
        # 进行反卷积
        # self.fc4 = nn.Linear(32, 1)
        self.fc4 = nn.Linear(23, 1)

    def encode(self, x):
        # 首先先修改输入的维度
        x = x.unsqueeze(1)  # b, 1, 3
        x = self.input(x)  # b, 1, 8
        x = self.conv1(x)  # b, 8, 3

        x = x.view(-1, 3 * 8)  # b, 24

        x = F.elu(self.out(x))
        # h1 = F.elu(self.fc1(x))  # (bathsize, 32)

        mu, sigma = x.chunk(2, dim=1)
        # return F.relu(self.fc21(x)), F.relu(self.fc22(x))  # (bathsize, 8)

        return mu, sigma

    def decode(self, z):
        # 首先修改维度
        z = z.unsqueeze(1)  # b, 1, 8

        z = F.relu(self.dconv1(z))  # b, 1, 23
        z = z.view(-1, 23)
        # h3 = F.relu(self.fc3(z))
        return self.fc4(z)

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码

        q = mu + logvar * torch.randn_like(logvar)

        x_hat = self.decode(q)
        # KL
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(logvar, 2) -
            torch.log(1e-8 + torch.pow(logvar, 2)) - 1
        ) / (128 * 3)

        # kld = 0.5 * torch.sum(
        #     torch.pow(mu, 2) +
        #     torch.pow(logvar, 2) -
        #     torch.log(1e-8 + torch.pow(logvar, 2)) - 1
        # )

        return x_hat, kld  # 解码，同时输出均值方差


# net = VAE()
# net.cuda()
#
# print(net)
# data = torch.randn(128, 3).cuda()
#
# out, _= net(data)
# # out = out.detach().numpy()
# print(out.shape)
class VAE_SINGLE(nn.Module):
    def __init__(self):
        super(VAE_SINGLE, self).__init__()

        # self.fc1 = nn.Linear(3, 32)

        self.input = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=8,
                      kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            # nn.MaxPool1d(kernel_size=2, padding=1),
            nn.LeakyReLU()
        )  # (b, 8, 3) ---> b, 24

        # self.fc21 = nn.Linear(32, 8)  # mean
        # self.fc22 = nn.Linear(32, 8)  # var

        # self.fc21 = nn.Linear(24, 8)  # mean
        # self.fc22 = nn.Linear(24, 8)  # var

        self.out = nn.Linear(24, 16)

        # self.fc3 = nn.Linear(8, 32)

        # 将b, 8--->, b, 1, 8
        self.dconv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)  # b, 1, 23
        # 进行反卷积
        # self.fc4 = nn.Linear(32, 1)
        self.fc4 = nn.Linear(23, 1)

    def encode(self, x):
        # 首先先修改输入的维度
        x = x.unsqueeze(1)  # b, 1, 3
        x = self.input(x)  # b, 1, 8
        x = self.conv1(x)  # b, 8, 3

        x = x.view(-1, 3 * 8)  # b, 24

        x = F.elu(self.out(x))
        # h1 = F.elu(self.fc1(x))  # (bathsize, 32)

        mu, sigma = x.chunk(2, dim=1)
        # return F.relu(self.fc21(x)), F.relu(self.fc22(x))  # (bathsize, 8)

        return mu, sigma

    def decode(self, z):
        # 首先修改维度
        z = z.unsqueeze(1)  # b, 1, 8

        z = F.relu(self.dconv1(z))  # b, 1, 23
        z = z.view(-1, 23)
        # h3 = F.relu(self.fc3(z))
        return self.fc4(z)

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码

        q = mu + logvar * torch.randn_like(logvar)

        x_hat = self.decode(q)
        # KL
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(logvar, 2) -
            torch.log(1e-8 + torch.pow(logvar, 2)) - 1
        ) / (128 * 3)

        # kld = 0.5 * torch.sum(
        #     torch.pow(mu, 2) +
        #     torch.pow(logvar, 2) -
        #     torch.log(1e-8 + torch.pow(logvar, 2)) - 1
        # )

        return x_hat, kld  # 解码，同时输出均值方差


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # self.fc1 = nn.Linear(3, 32)

        self.input = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=8,
                      kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            # nn.MaxPool1d(kernel_size=2, padding=1),
            nn.LeakyReLU()
        )  # (b, 8, 3) ---> b, 24

        self.e_out = nn.Linear(24, 8)

        # 将b, 8--->, b, 1, 8
        self.dconv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)  # b, 1, 23
        # 进行反卷积
        self.fc4 = nn.Linear(23, 1)

    def encode(self, x):
        # 首先先修改输入的维度
        x = x.unsqueeze(1)  # b, 1, 3
        x = self.input(x)  # b, 1, 8
        x = self.conv1(x)  # b, 8, 3

        x = x.view(-1, 3 * 8)  # b, 24

        x = F.elu(self.e_out(x))

        return x

    def decode(self, z):
        # 首先修改维度
        z = z.unsqueeze(1)  # b, 1, 8
        z = F.relu(self.dconv1(z))  # b, 1, 23
        z = z.view(-1, 23)
        return self.fc4(z)

    def forward(self, x):
        en_out = self.encode(x)  # 编码

        x_hat = self.decode(en_out)

        return en_out, x_hat  # 编码, 解码


class CAE_SINGLE(nn.Module):
    def __init__(self):
        super(CAE_SINGLE, self).__init__()

        # self.fc1 = nn.Linear(3, 32)

        self.input = nn.Sequential(
            nn.Linear(1, 8),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=8,
                      kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            # nn.MaxPool1d(kernel_size=2, padding=1),
            nn.LeakyReLU()
        )  # (b, 8, 3) ---> b, 24

        self.e_out = nn.Linear(24, 8)

        # 将b, 8--->, b, 1, 8
        self.dconv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)  # b, 1, 23
        # 进行反卷积
        self.fc4 = nn.Linear(23, 1)

    def encode(self, x):
        # 首先先修改输入的维度
        x = x.unsqueeze(1)  # b, 1, 3
        x = self.input(x)  # b, 1, 8
        x = self.conv1(x)  # b, 8, 3

        x = x.view(-1, 3 * 8)  # b, 24

        x = F.elu(self.e_out(x))

        return x

    def decode(self, z):
        # 首先修改维度
        z = z.unsqueeze(1)  # b, 1, 8
        z = F.relu(self.dconv1(z))  # b, 1, 23
        z = z.view(-1, 23)
        return self.fc4(z)

    def forward(self, x):
        en_out = self.encode(x)  # 编码

        x_hat = self.decode(en_out)

        return en_out, x_hat  # 编码, 解码


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # self.fc1 = nn.Linear(3, 32)

        self.input = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=8,
                      kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            # nn.MaxPool1d(kernel_size=2, padding=1),
            nn.LeakyReLU()
        )  # (b, 8, 3) ---> b, 24

        self.e_out = nn.Linear(24, 8)

        # 将b, 8--->, b, 1, 8
        self.dconv1 = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)  # b, 1, 23
        # 进行反卷积
        self.fc4 = nn.Linear(23, 1)

    def encode(self, x):
        # 首先先修改输入的维度
        x = x.unsqueeze(1)  # b, 1, 3
        x = self.input(x)  # b, 1, 8
        x = self.conv1(x)  # b, 8, 3

        x = x.view(-1, 3 * 8)  # b, 24

        x = F.elu(self.e_out(x))

        return x

    def decode(self, z):
        # 首先修改维度
        z = z.unsqueeze(1)  # b, 1, 8
        z = F.relu(self.dconv1(z))  # b, 1, 23
        z = z.view(-1, 23)
        return self.fc4(z)

    def forward(self, x):
        en_out = self.encode(x)  # 编码

        x_hat = self.decode(en_out)

        return en_out, x_hat  # 编码, 解码

