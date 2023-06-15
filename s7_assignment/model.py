import torch.nn as nn
import torch.nn.functional as F

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
         
    
