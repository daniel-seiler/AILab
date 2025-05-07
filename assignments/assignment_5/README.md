# Assignment 5

## Task
Train a text classification on the TweetEval emotion recognition dataset using LSTMs and GRUs.
- Follow the example described here. Use the same architecture, but:
    - only use the last output of the LSTM in the loss function
    - use an embedding dim of 128
    - use a hidden dim of 256.
- Use spaCy to split the tweets into words and remove all stop words (words like „the“, „a“, etc).
- Limit your vocabulary (i.e. the words that you converted to an index) to the most frequent 3000
words. Replace all other words with an placeholder index (e.g. 3001).
- Evaluate the accuracy on the test set.
- Note: If the training takes to long, try to use only a fraction of the training data.
- Do the same, but this time use GRUs instead of LSTMs.

## Bonus
- The training is quite slow, since we do not train in batches but line by line. Providing
the text lines in batches to the LSTM, will not work out of the box, because all
tensors in one batch need to have the same length.
- The DataLoader class can be created with a collate_fn function that can be
used to bring all lines in one batch to the same size:
    - Determine the longest tensor in the current batch.
    - Fill up all tensors that are shorter with an unused padding value (e.g. -1).
- Task: Put the data into a DataLoader, pad the lines in one batch to the same length
and train in batches on the GPU. Compare the training time to the line-wise training.