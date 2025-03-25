# Assignment 1

## Preparation:
1. Checkout the notebook 0_Simple_NN.py from [GitHub](https://github.com/pabair/ki-lab-ss23), which describes how to
setup a simple feedforward network on some fake data. Try to understand
everything and execute the code in a jupyter notebook.
2. Read through this blog post: https://nextjournal.com/gkoehler/pytorch-mnist
It shows how to work with data loaders, how to load the MNIST dataset and how
training is done in batches (using data loaders).
3. Read the famous blog post [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) to learn more
about tricks and best practices for model training

## Task:
1. Load the MNIST dataset into train and test data loaders. Use the same
parameters and apply the same transformations like described in the blog post.
2. Create a feedforward neural network consisting of an input layer, one hidden
layer of size 100 and an output layer (same structure as in 0_Simple_NN.py).
For training on the MNIST dataset you need to change the following:
    - Adjust the size of the input layer to be able to take in the MNIST data (hint: you
    must adjust the tensor format from the MNIST data into a flat structure).
    - Use log_softmax as activation function for the output layer (as in the blog).
    Note: Do not use a CNN like they do in the blog post! Use Relu as activation
    function for the hidden layer.
3. Train your network on the training data for 50 epochs using the negative log
likelihood loss (like in the blog). Create a plot of the training loss (like in the blog
but without the test loss).
4. Test the network on the MNIST test data and give out accuracy and loss.
5. Find out how the model can be trained on the GPU instead of the CPU. Compare the
training time between CPU and GPU. (Note: Do not except too much improvement on
this small data set).