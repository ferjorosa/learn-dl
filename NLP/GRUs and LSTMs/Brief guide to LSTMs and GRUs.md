## Brief guide to LSTMs and GRUs

### The problem: short-term memory

Recurrent neural networks (RNNs) suffer from short-term memory. If a sequence is long enough, they will have a hard time carrying information from earlier time steps to later ones. So if you are trying to process a paragraph of text to do predictions, RNNs may leave out important information from the beginning.

During back propagation, RNNs suffer from the vanishing gradient problem, where gradients shrink as they back propagate through time. If a gradient becomes extremely small, it doesn't contribute too much learning.

So in RNNs, small gradient updates stop learning. Those are usually produced by the earlier layers (those more far away from the end when backpropagating). Thus, RNNs can forget what it has seen in longer sequences (i.e., short-term memory).

### LSTMs and GRUs as a solution
**LSTMs** and **GRUs** were created as the solution to short-term memory. They have internal mechanism called "gates" that can regulate the flow of information. These gates can learn which data in a sequence is important to keep or throw away. By doing that, it can pass relevant information down the long chain of sequences to make predictions.




<img src="images/2layer_rnn.png" width="600">