# Assignment 3
## Task:
1. Train the same network as in the PyTorch CNN tutorial.
2. Change now the network architecture as follows and train the network:
    1. Conv layer with 3x3 kernel and depth = 8, ReLu activation
    2. Conv layer with 3x3 kernel and depth = 16, ReLu activation
    3. Max pooling with 2x2 kernel
    4. Conv layer with 3x3 kernel and depth = 32, ReLu activation
    5. Conv layer with 3x3 kernel and depth = 64, ReLu activation
    6. Max pooling with 2x2 kernel
    7. Fully connected with 4096 nodes, ReLu activation
    8. Fully connected with 1000 nodes, ReLu activation
    9. Fully connected with 10 nodes, no activation 15
    input
    3x3 conv, 8
    3x3 conv, 16
    pool 2x2
    3x3 conv, 32
    3x3 conv, 64
    pool 2x2
    FC, 4096
    FC, 1000
    FC, 10
    output
3. Run the training on the GPU and compare the training time to CPU.
4. Log the training loss in tensorboard and also the graph of the network.
5. Change the test metric as follows: A prediction is considered „correct“ if the true
label is within the top three outputs of the network. Print the accuracy on the test
data (with respect to this new definition).
6. Randomly take 5 examples on which the network was wrong on the test data
(according to the new definition of correct) and plot them to tensorboard together
with the true label.
7. Show the tensor board widget at the end of your notebook.
Bonus: See if you can get better by using a deeper network (or another architecture).