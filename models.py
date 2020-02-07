## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #Init: Have 5 conv, for each conv have one pooling layer
        
        # From http://cs231n.github.io/convolutional-networks/
        # Summary. To summarize, the Conv Layer:
        # Accepts a volume of size W1×H1×D1
        # Requires four hyperparameters:
        # Number of filters K,
        # their spatial extent F,
        # the stride S,
        # the amount of zero padding P.
        # Produces a volume of size W2×H2×D2 where:
        # W2=(W1−F+2P)/S+1
        # H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
        # D2=K
        # With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
        # In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th 
        # filter over the input volume with a stride of S, and then offset by d-th bias.
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
       
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        # after another pool layer this becomes (64, 54, 54); 
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (128, 52, 52)
        # after another pool layer this becomes (128, 26, 26); 
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        # 4th conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12); 
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 5th conv layer: 256 inputs, 512 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        # the output tensor will have dimensions: (512, 10, 10)
        # after another pool layer this becomes (512, 5, 5); 
        self.conv5 = nn.Conv2d(256, 512, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # 512 outputs * the 5*5 filtered/pooled map size
        # 136 output channels (for the 10 classes)
        #self.fc1 = nn.Linear(128*24*24, 136)
        self.fc1 = nn.Linear(512*6*6, 1024)
        
        self.fc2 = nn.Linear(1024, 136)
        
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.6)
         
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        DEBUG_SHAPE = False
        if DEBUG_SHAPE: 
            print("1. ",x.shape) 
        x = self.pool1(F.relu(self.conv1(x)))
        if DEBUG_SHAPE: print("2. ",x.shape) 
     
        x = self.pool2(F.relu(self.conv2(x)))
        if DEBUG_SHAPE: print("3. ", x.shape) 
    
        x = self.pool3(F.relu(self.conv3(x)))
        if DEBUG_SHAPE: print("3a. ", x.shape) 
        x = self.dropout1(self.pool4(F.relu(self.conv4(x))))
        if DEBUG_SHAPE: print("3b. ", x.shape) 
        x = self.dropout2(self.pool5(F.relu(self.conv5(x))))
        if DEBUG_SHAPE: print("3c. ", x.shape,x.size(0))
        
       
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        if DEBUG_SHAPE: print("4. " ,x.shape) 
        x = self.dropout3(F.relu(self.fc1(x)))
        if DEBUG_SHAPE: print("5. " ,x.shape) 
        x = self.fc2(x)
        
    
        # one linear layer
        #x = F.relu(self.fc1(x))  
        #x = F.relu(self.fc1(self.dropout(x)))
        #if DEBUG_SHAPE: print("6. " ,x.shape) 
        #x = self.fc2(self.dropout(x))
        # a softmax layer to convert the 10 outputs into a distribution of class scores
        
        #x = F.log_softmax(x, dim=1)  
        
        #if DEBUG_SHAPE: print("7. " ,x.shape) 
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
