import torch
import torch.nn as nn

# Performs two residual blocks and one downsample
# A residual block consists of a [convolution, batchnorm, relu] with a skip connection
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()

        self.res0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        res0_in = x
        res0_out = self.res0(res0_in)

        res1_in = torch.add(res0_out, res0_in)
        res1_out = self.res1(res1_in)

        downsample_in = torch.add(res1_out, res1_in)
        downsample_out = self.downsample(downsample_in)
        return downsample_out

# Performs one upsample and two residual blocks
# A residual block consists of a [convolution, batchnorm, relu] with a skip connection
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        upsample_in = x
        upsample_out = self.upsample(upsample_in)

        res0_in = upsample_out
        res0_out = self.res0(res0_in)
        res1_in = torch.add(res0_out, res0_in)
        res1_out = self.res1(res1_in)
        return torch.add(res1_in, res1_out)

# Performs three residual blocks
# A residual block consists of a [convolution, batchnorm, relu] with a skip connection
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()

        self.res0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        res0_in = x
        res0_out = self.res0(res0_in)

        res1_in = torch.add(res0_in, res0_out)
        res1_out = self.res1(res1_in)
        
        res2_in = torch.add(res1_in, res1_out)
        res2_out = self.res2(res2_in)
        return res2_out

# UNet network comprised of encoders, bottleneck, and decoders
class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # model parameters
        down_channels = 48
        up_channels = 48
        kernel_size = 3

        # one extra input convolution to get the channel dimension to the correct size for the res blocks
        self.input_block = nn.Conv2d(
            in_channels=in_channels, # network input dimension
            out_channels=down_channels,
            kernel_size=kernel_size,
            padding="same"
        )

        # NOTE: for symmetry, down blocks are numbered 0, 1, 2... but up blocks are numbered 2, 1, 0...
        self.down0 = EncoderBlock(
            in_channels=down_channels,
            out_channels=down_channels,
            kernel_size=kernel_size
        )
        self.down1 = EncoderBlock(
            in_channels=down_channels,
            out_channels=down_channels,
            kernel_size=kernel_size
        )
        self.down2 = EncoderBlock(
            in_channels=down_channels,
            out_channels=down_channels,
            kernel_size=kernel_size
        )
        self.bottleneck = Bottleneck(
            in_channels=down_channels,
            out_channels=down_channels,
            kernel_size=kernel_size
        )
        self.up2 = DecoderBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            kernel_size=kernel_size
        )
        self.up1 = DecoderBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            kernel_size=kernel_size
        )
        self.up0 = DecoderBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            kernel_size=kernel_size
        )

        # one extra input convolution to get the channel dimension to the correct size for output 
        self.output_block = nn.Conv2d(
            in_channels=up_channels,
            out_channels=3, # network output dimension
            kernel_size=kernel_size,
            padding="same"
        )

    def forward(self, x):
        xin = self.input_block(x)

        # encoder
        x0 = self.down0(xin) 
        x1 = self.down1(x0) 
        x2 = self.down2(x1) 
        
        x3 = self.bottleneck(x2) 

        # decoder
        xu2 = self.up2(torch.add(x3, x2))
        xu1 = self.up1(torch.add(xu2, x1))
        xu0 = self.up0(torch.add(xu1, x0))

        xout = self.output_block(xu0)

        # might as well sigmoid the final result since we want it to be between 0 and 1 always
        xout = torch.sigmoid(xout)
        return xout

def LossFn(y, target, prev_y, prev_target):
    l1_weight = 0.8
    temporal_l1_weight = 0.2

    l1_loss = l1_weight * nn.functional.l1_loss(y, target)
    temporal_l1_loss = temporal_l1_weight * nn.functional.l1_loss(
        torch.subtract(y, prev_y),
        torch.subtract(target, prev_target)
    )
    return l1_loss + temporal_l1_loss