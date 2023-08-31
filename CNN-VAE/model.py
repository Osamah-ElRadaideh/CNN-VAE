import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,act='relu'):
        super().__init__()
        assert act.lower() in ['relu', 'none'], f'act must be either relu or none, got {act} instead'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act =='relu':
            return(self.relu(x))
        else:
            return x 

        
class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,act='relu', padding=0):
        super().__init__()
        assert act.lower() in ['relu', 'sigmoid'], f'act must be either relu or sigmoid, got {act} instead'
        self.act = act
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if self.act =='relu':
            return(self.relu(x))
        else:
            # return x 
            return(self.sigmoid(x))        
        


class Encoder(nn.Module):
    def __init__(self, channel_size=64, in_channels = 3) :
        super().__init__()
        self.conv1 = down_block(in_channels, channel_size)
        self.conv2 = down_block(channel_size, 2 * channel_size)
        self.conv3 = down_block(2 * channel_size, 4 * channel_size)
        self.conv4 = down_block(4 * channel_size, 8 * channel_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)
    
class Decoder(nn.Module):
    def __init__(self,channel_size=512, out_channels=3):
        super().__init__()
        self.conv1 = up_block(channel_size, channel_size // 2)
        self.conv2 = up_block(channel_size // 2, channel_size // 4)
        self.conv3 = up_block(channel_size // 4, channel_size // 8, padding=1)
        self.conv4 = up_block(channel_size // 8, out_channels,act='sigmoid')
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)
    

class VAE(nn.Module):
    def __init__(self, channel_size=64, in_channels=3,  mu_dim=2048, var_dim=64):
        super().__init__()
        self.encoder = Encoder(channel_size, in_channels)
        self.decoder = Decoder(channel_size * 8, in_channels)
        self.fc1 = nn.Linear(mu_dim, var_dim)
        self.fc2 = nn.Linear(mu_dim, var_dim)
        self.fc3 = nn.Linear(var_dim, mu_dim)
        self.flatten = nn.Flatten()
        self.channel_size = channel_size
    def rep(self, mu, var):
        #reparameterization, i.e sampling
        std = var.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    def get_params(self, x):
        x = self.flatten(x)
        mu, var = self.fc1(x), self.fc2(x)
        z = self.rep(mu, var)
        return z, mu, var
    
    def encode(self, x):
        encoded = self.encoder(x)
        z, mu, var = self.get_params(encoded)
        return z, mu, var
    
    def decode(self, x):
        x = self.fc3(x)
        x = x.view(x.shape[0], self.channel_size * 8, 2, 2) 
        x = self.decoder(x)
        return x
        
        
    def forward(self, x):
        enc, mu, var = self.encode(x)
        dec = self.decode(enc)
        return dec, mu, var
    




def vae_loss(decoded, orig, mu, var):
    loss = F.l1_loss(orig, decoded,)

    KLD = -0.5 * torch.mean(1 + var - mu.pow(2) - var.exp())

    return loss + KLD