
import torch
import torch.nn as nn
from   torch.autograd import Variable
##########################################################

#Linear regression model
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

##########################################################

#Two possible Deep neural network models; in the main, neural1 was eventually used

class neural(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(inputSize, 128)
        self.batchnorm1    = nn.BatchNorm1d(128)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(128, outputSize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.batchnorm1(x)
        x = self.sigmoid(x)
        x = self.output(x)

        return x
class neural_1(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.hidden1 = nn.Linear(inputSize, 128)
        self.batchnorm1    = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 64)
        self.batchnorm2    = nn.BatchNorm1d(64)
        self.dropout=nn.Dropout(p=0.000001)
        self.output = nn.Linear(64, outputSize)
        self.batchnorm3    = nn.BatchNorm1d(64)
        self.hidden3 = nn.Linear(64,32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.hidden2(x)
        x = self.batchnorm2(x)
        #x = self.dropout(x)
        #x = self.hidden3(x)
        #x = self.batchnorm3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        # x = self.softmax(x)

        return x
