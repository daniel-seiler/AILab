# Assignment 4
1. Check out the torchvision datasets of PyTorch and decide one dataset that you want to use
(please do not use: CIFAR, ImageNet, FashionMNIST or eMNIST).
2. Show some example images of the dataset in the notebook and print the dataset size.
3. Design a CNN to predict on the dataset. Use a similar architecture like last time, but this time
also include batch normalization layers. See here for a short description.
4. Train the model on the dataset and measure the accuracy on hold out test data.
5. Now use transfer learning to use a pre-trained ResNet18 on the dataset as follows:
    1. ResNet18 as fixed feature extractor.
    2. ResNet18 finetuned on the training data (remember to adapt the learning rate).
6. Repeat step 4 but now use EfficientNet_B5 instead of RestNet18.
7. Compare the accuracy of the different approaches on the test data and print out the training
times for each approach.