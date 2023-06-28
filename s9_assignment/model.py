import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary



dropout_value = 0.1
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, groups=160, bias=False), #Depthwise Separable Convolution
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, groups=160, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
        ) 
            
        # CONVOLUTION BLOCK 3       
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=160, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, groups=160, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(160),
            nn.Conv2d(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) 

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
        )

        # TRANSITION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
        )

            
        # CONVOLUTION BLOCK 4       
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, groups=320, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Conv2d(in_channels=320, out_channels=132, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(132),
            
        ) 

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 132, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(132),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) 

        self.linear = nn.Linear(132, 10)
        

    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = x2 + x1

        x4 = self.convblock3(x3)

        x5 = self.convblock4(x4)
        x6 = x5 + x4

        x7 = self.convblock6(x6)

        x8 = self.convblock7(x7)
        x9 = x8 + self.shortcut1(x7)

        x10 = self.convblock8(x9)

        x11 = self.convblock9(x10)
        x12 = x11 + self.shortcut2(x10)


        out = self.gap(x12)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return F.log_softmax(out)



'''
Assignment 8
'''

dropout_value = 0.1


class cnn_norm(nn.Module):
    def __init__(self, use_batch_norm=False, use_layer_norm=False, use_group_norm=False, num_groups=2):
        super(cnn_norm, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups

# ***************** BATCH NORMALIZAION ############################
        if self.use_batch_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          ) # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 1, RF=44


# ***************** LAYER NORMALIZAION ############################
        elif self.use_layer_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          ) # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 1, RF=44




# ***************** GROUP NORMALIZAION ############################
        elif self.use_group_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16),
              nn.Dropout(dropout_value)
          ) # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(24),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Dropout(dropout_value)
          ) # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          ) # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 1, RF=44



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



'''
Assignment 7
'''
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),    # 28x28x1 > 28x28x8   : RF 3x3
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),    # 28x28x8 > 28x28x16   : RF 5x5
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),   # 28x28x16 > 28x28x32  : RF 7x7
            nn.ReLU(),
            nn.Conv2d(32,64, 3, padding=1),  # 28x28x32 > 28x28x64 : RF 9x9
            nn.ReLU(),
            nn.MaxPool2d(2,2),                # 28x28x16 > 14x14x16 : RF 10x10
            nn.Conv2d(64, 128, 3, padding=1),  # 14x14x64 > 14x14x128 : RF 14x14
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  # 14x14x128 > 14x14x128 : RF 18x18
            nn.ReLU(),
            nn.MaxPool2d(2,2),                # 14x14x128 > 7x7x128   : RF 20x20
            nn.Conv2d(128, 10, 3),             # 7x7x128 > 5x5x10     : RF 28x28
            nn.AvgPool2d(5, 2),               # 5x5x10 > 1x1x10     : RF 32x32
        )
 
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x
    
## The below model reduces the parameters by reducing the number of input and output channels and we have batch normalisation introduced    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),    # 28x28x1 > 28x28x4   : RF 3x3
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1),    # 28x28x4 > 28x28x8   : RF 5x5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1),   # 28x28x8 > 28x28x12  : RF 7x7
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding=1),  # 28x28x12 > 28x28x16 : RF 9x9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),                # 28x28x16 > 14x14x16 : RF 10x10
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14x16 > 14x14x32 : RF 14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),  # 14x14x32 > 14x14x32 : RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),                # 14x14x32 > 7x7x32   : RF 20x20
            nn.Conv2d(32, 10, 3),             # 7x7x32 > 5x5x10     : RF 28x28
            nn.AvgPool2d(5, 2),               # 5x5x10 > 1x1x10     : RF 32x32
        )
 
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

## The model 3 we have dropout introduced as 0.1 for overcoming the overfitting
dropout_value=0.05
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),    # 28x28x1 > 28x28x4   : RF 3x3
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(dropout_value),
            nn.Conv2d(4, 8, 3, padding=1),    # 28x28x4 > 28x28x8   : RF 5x5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 12, 3, padding=1),   # 28x28x8 > 28x28x12  : RF 7x7
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 16, 3, padding=1),  # 28x28x12 > 28x28x16 : RF 9x9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2,2),                # 28x28x16 > 14x14x16 : RF 10x10
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14x16 > 14x14x32 : RF 14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1),  # 14x14x32 > 14x14x32 : RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2,2),                # 14x14x32 > 7x7x32   : RF 20x20
            nn.Conv2d(32, 10, 3),             # 7x7x32 > 5x5x10     : RF 28x28
            nn.AvgPool2d(5, 2),               # 5x5x10 > 1x1x10     : RF 32x32
        )
 
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

##On top of the drop out , we are bringing the 1x1 convolution for channel reduction. The number of parameters is now optimised
class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  #28x28x1 > 28x28x8 : RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 12, 3, padding=1), # 28x28x8 > 28x28x12 : RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(12), 
            nn.Dropout(dropout_value),
            #transition block
            nn.Conv2d(12, 6, 1),            # 28x28x12 > 28x28x6  : RF: 5x5
            nn.MaxPool2d(2,2),              # 28x28x6 > 14x14x6   : RF: 6x6
            nn.Conv2d(6, 12, 3),            # 14x14x6  > 12x12x12 : RF: 10x10
            nn.ReLU(),
            nn.BatchNorm2d(12), 
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3),           # 12x12x12 > 10x10x12 : RF: 14x14
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding= 1),# 10x10x12  > 10x10x12 : RF: 18x18
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2,2),               # 10x10x12 > 5x5x12    : RF: 20x20
            nn.Conv2d(12, 12, 3),            # 5x5x12 > 3x3x12      : RF: 28x28
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding= 1),# 3x3x12 > 3x3x12      : RF: 36x36
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value),
            nn.AvgPool2d(3, 2),              # 3x3x12 > 1x1x12 : RF: 40x40
            nn.Conv2d(12, 10 , 1)            # 1x1x12 > 1x1x10 : RF: 40x40
        )
     
 
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
'''
Assignment 5
'''
    
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
    
         
    
