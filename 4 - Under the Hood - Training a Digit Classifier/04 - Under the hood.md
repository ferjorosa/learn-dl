## 04 - UNDER THE HOOD: TRAINING A DIGIT CLASSIFIER

A PyTorch tensor is nearly the seam thing as a Numpy array, but with an additional restriction that unlocks additional capabilities. It is the same in that it too is a multidimensional table of data with items all of the same type. However, the restriction is that **a tensor has to use a single numeric type for all its components**, it cannot use any old type.

As a result, a tensor is not as flexible as a genuine array of arrays and cannot be jagged (i.e., to be composed of arrays of different lengths. For example [[1,2,3], [4,5]]). **A tensor is always a regularly-shaped multidimensional rectangular structure**.

The vast majortiy of methods and operators supported by Numpy on these rectangular structures are also supported by PyTorch. However, PyTorch tensors have additional capabilities. One of them is that these structures can live on the GPU and thus be executed much faster.

### Tensor basics
The <code>stack()</code> method allows us to combine matrices (2d arrays) into tensors. For example, if we stack 2 matrices of 3x3, we would end up with a 2x3x3 tensor. That is a rank 3 tensor because it considers 3 dimensions.

When substracting tensors (e.g., a tensor of 2x3x3 minus a 3x3 matrix), we may be interested in obtaining the mean. In that case, we would need to indicate the relevant dimension or PyTorch will consider all of them. In this specific case, we would be interested in the last two dimensions: <code>mean((-1,-2))</code>. For this example, it is equivalent to <code>mean((1,2))</code> because the rank of the tensor is 3. 
* In the first case, we are selecting the last element (-1) and the second last element (-2). Similar notation to python lists.
* In the second case, we are selecting the second element (1) and the third element (2). Similar notation to python lists.

### Stochastic gradient descent
Following the digit classifier example, instead of trying to find the similarity between an image and an "ideal" image, we could instead loot at each individual pixel and come up with a set of weights for each, such that the highest weights ar associated with those pixels most likely to be black for a particular category.

Here is a very simplified process of how we can learn these weights:
1. Initialize the weights.
2. For each image, use the weights to predict wether it appears to be a "3" or a "7".
3. Based on these predictions, calculate how good the model is (its **loss**)
4. Calculate the gradient, which measure for each weight how changing the weight would affect the loss
5. **Step** (that is, change) all the weights based on that calculation.
6. Go back to step 2 and repeat.

This process is repeated until we want to stop (e.g., after multiple iterations or after the loss change is small).

In the digit classifier example, we could repeat this process until the accuracy of the model starts getting worse.

### Calculating gradients
In math, the "gradient" of a function is just another function, more specifically, its derivative. However, in deep learning, "gradient" usually refers to the **value of a function's derivative at a particular argument value**.

As an example, consider a quadratic function:

```
def f(x): return x**2
```

If we want to consider a simple tensor from which we want its gradients, we use:

```
xt = tensor(3.).required_grad_()
```

PyTorch will then automatically notice that a gradient may be required:
```
yt = f(xt)
yt ----> tensor(9., grad_fn=<PowerBackward0)
```

We can tell PyTorch to calculate the gradients easily:

```
yt.backward()
```

We can now view gradients by checking the <code>grad</code> attribute of our tensor:

```
xt.grad -----> tensor(6.)
```

The derivative of x^2 y 2x, so this coincides!


Note that the gradients tell us only the slope of our function. They don't tell us exactly how far to adjust the parameters. But they do give us some idea of how far: if the slope is very large, that may suggest that we have more adjusments to do, whereas if the slope is very small, that may suggest that we are close to the optimal value.

### Stepping with a learning rate
Deciding how to change our parameters based on the values of the gradients is an important part of the deep learning process. Nearly all approaches start with the basic notion of multiplying the gradient by some small number, called the **learning rate** (usually between 0.001 and 0.1).

### The MNIST loss function
For the representation of our model, we are going to use a linear regression <code>y = wx + b</code>, where <code>w</code> are the weights of our model and <code>b</code> is the intercept.

Now, given that our objective with the MNIST problem is to classify 2 types of numbers (i.e., threes and sevens), we may not be able to do this with a regression model. The main problem resides in the fact that the linear regression returns a continuous value. We could transform it into a categorical binary value by establishing a threshold. However, we have another problem, where should we place that threshold? We can solve this second problemby using another function that properly transforms our linear regression model into a [0,1] value: the **sigmoid** function.

### Why use a loss function when we have the accuracy metric?
The key difference is that a loss function is used to drive automated learning and accuracy (and similar metrics) are used in human understdanding.
* To drive automated learning, the loss must be a function that has a **meaningful derivative**.
* Human metrics are the numbers we should care about rather than the loss when judging the performance of a model.









