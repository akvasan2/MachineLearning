# Deep neural network fit to a dataset of textual descriptors of different neural network architectures

Each architecture was trained on the Cifar-10 dataset and represent error for training different models for image detection. 

Each model has: 
    >200 features,
    and two sets of labels.

The deep neural network consists of: 

    2 hidden layers,
    a sigmoid activation function,
    batch normalization,
    
    Gradient method: Adam, learning rate = 0.01
    Loss Function: SmoothL1Loss
