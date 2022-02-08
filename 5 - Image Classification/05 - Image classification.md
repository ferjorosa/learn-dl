## 5 - IMAGE CLASSIFICATION

### Presizing

We need our images to have the same dimensions, so that they can collate into tensors. We also want to minimize the number of distinct augmentation computations to perform.

The challenge is that, if performed after resizing down to the augmented size, various common data augmentation transforms might introduce spurious empty zones, degrade data or both.

To work around these challenges, presizing adopts two strategies:

1. Resize images to relatively "large" dimensions so that they have spare margin to allow further augmentation transforms on their inner regions without creating empty zones.
2. Compose all of the common augmentation operations (including a resize to the final target size) into one, and perform the combined operation on the GPU only once at the end of processing, rather than performing the operations individually and interpolating multiple times

The second strategy is very important because moving operations from the CPU to the GPU has a intrinsic computational cost.

### Cross-entropy loss
Cross-entropy loss is a loss function for classification that is similar to the one used in Chapter 4m but with the following advantages:

1. It works even when the predictive variable has more than two categories.
2. It results in faster and more reliable training.

To understand how cross-entropy loss works for dependent (i.e., predictive) variables with more than 2 categories, we have to first understand the associated activation function: the **softmax**.

### Softmax activation function
The softmax function is similar to the sigmoid function but it allows more than two categories.

In the MNIST example, we considered a model with a single column of activations for predicting the numbers 3 or 7. However, if we want to predict multiple categories, this will not be enough. We will need an activation per category.

In this context, **we cannot use the sigmoid function because the final result would not add up to 1**. For Example:

```
[[-2.24, 0.38, -0.67], [1.50, -2.30, 0.92]] ----> [[0.66, 0.56, 0.20], [0.15, 0.98, 0.87]]
```

For comparison, here are the formulas of the sigmoid and softmax functions:

```
def sigmoid(x): return 1 / 1+exp(x)
def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
```

Softmax is the multi-category equivalent of sigmoid. We have to use it anytime we have more than two categories and the probabilities of the categories must add up to 1.

We could create other functions that have the properties with the properties of returning values in [0,1] and summing up to 1. However, no other function has the same relationship with the sigmoid function, which we have seen is smooth and symmetric. In addition, we will see that the softmax function works well with the cross-entropy loss (i.e., log-likelihood).

What does the **softmax** function do in practice? Taking the exponential ensures that all numbers are positive, and then dividing by the sum ensures that we are going to have a series of numbers that add up to 1. The exponential also has a nice property: if one of the numbers in our activations X is slightly bigger than the ohers, the exponential will amplify this, which means that in the softmax, the number will be closer to 1. Intuitively, the softmax function really wants to pick one class among the others. So it is ideal for training a classifier when we know each picture has a definite label.

**Note that this approach may be less ideal during inference, as we want our model to sometimes tell us that if it does not recognize any of the classes that it has seen during training and not pick a class because it has a slightly bigger activation score. In this case, it might be better to train a model using multiple binary output columns, each using a sigmoid function**.

### Log-likelihood

Just as we moved from sigmoid function to softmax, we need to extend the loss function to work with more than just binary classification. The error estimation is simple. Since we have a vector with the Y label, for each data instance, the error would be 1 - prediction for the specific label. For example:

```
y = [0,2,0,1] # 4 class values from 3 possible categories
predictions = [
[0.7, 0.2, 0.1], 
[0.5, 0.4, 0.1], 
[0.1, 0.8, 0.1], 
[0.4, 0.2, 0.4]] ------> [0,7, 0.1, 0.1, 0.2] # Probabilities assigned by the model for each data instance to the true class value

error = (1 - 0.7) + (1 - 0.1) + (1 - 0.1) + (1 - 0.2)
```

**Note:** From my experience in probability, the error should be the multiplication of those values, by I am not 100% sure. Either way, we are going to sum the log-likelihoods...

### Taking the log
The problem with directly using probabilties is thath the model will not care wether it predicts 0.99 or 0.999. Indeed, those numbers are very close together, but in another sens, 0.999 is 10 more confident than 0.99. So, we want to transform our numbers between 0 and 1 to instead a range between netive infinity and infinity. There is a mathematical function that does exactly this: **the logarithm**.

Logarithms are widely used for multiplying very big or very small numbers because that multiplication can be transformed into a sum of logarithms.

Taking the mean of the sum of log-probabilities give us the negative log-likelihood loss. In Pytorch <code>nll_loss</code> assumes that you already took the log of the softmax, so it doesn't do the logarithm for you. In order to do it, we can use <code>log_softmax</code>, which combines  both functions in a fast an accurate way.

When we first take the softmax and then the log-likelihood of that, the resulting combination is called **cross-entropy loss**. In PyTorch, it is available as a class and as a function.
* class:
```
loss_func = nn.CrossEntropyLoss()
loss_func(acts,targ)
```
* function:
```
F.crossEntropyLoss(acts, targ)
```

An interesting feature about cross-entropy loss appears when we consider its gradient (see page 203). In summary, it results in a linear gradient, which leads to smoother training of models since we won't see sudden jumpts or exponential increases in gradients. **This is the same as mean-square error in regression**.

### Improving our model

We will now look at a range of techniques to improve the training of our model. While doing so, we will dive a little more on transfer learning and how to fine-tune our pretrained model as best as possible, without breaking pretrained weights.

#### The learning rate finder

One of the most important things we can do when training a model is to make sure that we have the right learning rate.

* If it is **too low**, the model may take many, many epochs to learn, and it may also mean overfitting problems. Since each complete pass through the data is a chance for our model to memorize it.
* If it is **too high**, the model may be unable to reach a good optimization point, since the gradient will step too far every time.

In 2015, Leslie Smith proposed the **learning rate finder** to find the perfect learning rate. The idea is simple but effective:
1. Start with a very, very small learning rate.
2. Use that for one-batch and record the losses afterward.
3. Increase the learning rate by a certain percentage (e.g., by doubling it), do another mini-batch, track the loss, and increase the learning rate again.
4. We keep doing this until the loss gets worse. This is the point we know we have gone too far, and a select a lower learning rate.

#### Unfreezing and transfer learning

We now know that a deep neural network usually consists of many linear layers with a nonlinear activation function between each pair followed by one or more final layers with an activation function such softmax ath the very end.

The final layer uses a matrix with enough columns such that the output size is the same as the number of classes in our model (assuming classification). This final layer was specifically designed to classify the categories in the original pretraining dataset. When doing transfer learning, we remove it and replace it with a new layer with the correct number of outputs for the desired task.

This newly added linear layer will have entirely random weigths and thus we need to fine-tune it.

Our challenge when fine-tuning is to replace the random weights with  others that correctly achieve our desired task. A simple trick can allow this to happen: **tell the optimizer to update the weights in only those randomly added final layers. Don't change the rest of the neural network at all**. This is called freezing those pretrained layers.

When we call the <code>fine_tune()</code> method, fastai does two things:
* Trains the randomly added layers for one epoch, with all the other layers frozen.
* Unfreezes all the layers and trains them for the number of epochs requested.




