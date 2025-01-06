import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
            padding = 0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size=5, 
            stride = 1,
            padding =2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size=3, 
            stride = 1,
            padding =1     
        )

        self.conv4 = nn.Conv2d(
            in_channels = 384,
            out_channels = 384,
            kernel_size=3, 
            stride = 1,
            padding =1     
        )

        self.conv5 = nn.Conv2d(
            in_channels = 384,
            out_channels = 256,
            kernel_size=3, 
            stride = 1,
            padding =1     
        )

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(
            in_features = 9216,
            out_features = 4096
        )
        self.dropout1 = nn.Dropout(0.5)


        self.fc2 = nn.Linear(
            in_features = 4096,
            out_features = 4096
        )
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(
            in_features = 4096,
            out_features = 1
        )


    def forward(self, image):
        bs, c, h, w = image.size()
        print("Initial shape", image.shape)   
        x = F.relu(self.conv1(image))   # 6, 96, 55, 55
        print("shape after conv1", x.shape)  
        x = self.pool1(x)    # 6, 96, 27, 27
        print("shape after pool1", x.shape)

        x = F.relu(self.conv2(x))    # 6, 256, 27, 27
        print("shape after conv2", x.shape)
        x = self.pool2(x)    # 6, 256, 13, 13
        x = F.relu(self.conv3(x)) # 6, 384, 13, 13
        x = F.relu(x + self.conv4(x)) # 6, 384, 13, 13
        x = F.relu(self.conv5(x)) # 6, 256, 13, 13

        x = self.pool3(x)   # 6, 256, 6, 6

        x = x.view(bs, -1)  # 6, 256*6*6 = 9216
        x = F.relu(self.fc1(x))  # 6, 4096
        x = self.dropout1(x)     # 6, 4096 

        x = F.relu(self.fc2(x))  # 6, 4096
        x = self.dropout2(x)     # 6, 4096 
        x = F.relu(self.fc3(x))  # 6, 1

        # x = torch.sigmoid(x, axis=1)  # 6, 1
        return x
        



