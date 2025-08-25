from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            channel_size*4, channel_size*8,
            kernel_size=4, stride=1, padding=1,
        )
        self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)

    def forward(self, x):
        #print("Layer 0:",x.shape)
        x = F.leaky_relu(self.conv1(x))
        #print("Layer 1:", x.shape)
        x = F.leaky_relu(self.conv2(x))
        #print("Layer 2:", x.shape)
        x = F.leaky_relu(self.conv3(x))
        #print("Layer 3:", x.shape)
        x = F.leaky_relu(self.conv4(x))
        #print("Layer 4:", x.shape)
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        #print("Layer 5:", x.shape)
        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
        self.bn0 = nn.BatchNorm2d(channel_size*8)
        self.bn1 = nn.BatchNorm2d(channel_size*4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size*8, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size*4, channel_size*2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size*2, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.sigmoid(g)
