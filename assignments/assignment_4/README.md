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

# Results

* Device: T4 GPU (colab)
* Batch size: 64
* Epochs: 5

| Model              |                                                                        Test Loss                                                                        |                                                                      Top3 Accuracy                                                                      |                                                                       Train Loss                                                                        |
|:-------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|
| SimpleNet          | ![Screenshot 2025-04-21 at 18-46-42 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-42%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-46-49 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-49%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-46-56 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-56%20assignment_4.ipynb%20-%20Colab.png) |
| ResNet18           | ![Screenshot 2025-04-21 at 18-45-37 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-45-37%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-45-44 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-45-44%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-45-51 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-45-51%20assignment_4.ipynb%20-%20Colab.png) |
| ResNet18 TL        | ![Screenshot 2025-04-21 at 18-46-14 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-14%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-46-24 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-24%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-46-33 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-46-33%20assignment_4.ipynb%20-%20Colab.png) |
| EfficientNet B5    | ![Screenshot 2025-04-21 at 18-44-47 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-44-47%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-44-55 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-44-55%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-45-03 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-45-03%20assignment_4.ipynb%20-%20Colab.png) |
| EfficientNet B5 TL | ![Screenshot 2025-04-21 at 18-54-13 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-54-13%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-54-21 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-54-21%20assignment_4.ipynb%20-%20Colab.png) | ![Screenshot 2025-04-21 at 18-54-28 assignment_4.ipynb - Colab.png](img%2FScreenshot%202025-04-21%20at%2018-54-28%20assignment_4.ipynb%20-%20Colab.png) |

| Model              | Top3 Accuracy | Training Time |
|--------------------|---------------|---------------|
| SimpleNet          | 95.02%        | 1.51 minutes  |
| ResNet18           | 97.43%        | 1.51 minutes  |
| ResNet18 TL        | 96.00%        | 1.39 minutes  |
| EfficientNet B5    | 94.78%        | 5.54 minutes  |
| EfficientNet B5 TL | 86.61%        | 2.46 minutes  |