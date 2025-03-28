# Assignment 2
## Task:
1. Write a custom dataset class for the titanic data (see the data folder on GitHub).
Use only the features: "Pclass", "Age", "SibSp", "Parch", „Fare“, „Sex“, „Embarked“.
Preprocess the features accordingly in that class (scaling, one-hot-encoding, etc) and
split the data into train and validation data (80% and 20%). The constructor of that class
should look like this:
titanic_train = TitanicDataSet('titanic.csv', train=True)
titanic_val = TitanicDataSet('titanic.csv', train=False)
2. Build a neural network with one hidden layer of size 3 that predicts the survival of the
passengers. Use a BCE loss (Hint: you need a sigmoid activation in the output layer).
Use a data loader to train in batches of size 16 and shuffle the data.
3. Evaluate the performance of the model on the validation data using accuracy as metric.
4. Create the following plot that was introduced
in the lecture:
5. Increase the complexity of the network by
adding more layers and neurons and see if
you can overfit on the training data.
6. Try to remove overfitting by introducing a
dropout layer.