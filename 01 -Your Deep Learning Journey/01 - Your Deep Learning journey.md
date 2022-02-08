## 1 - YOUR DEEP LEARNING JOURNEY

The FastAi function <code>cnn_learner</code> has a "pretrained"  parameter that defaults to true, which sets the weight of our model to values that have already been trained by experts to recognize a thousand different categories across 1.3 million photos (using ImageNet).

A model that has weights that have already been trained on another dataset is called a pretrained model. In most cases, it is adviseable to use pretrained models because ven before showing any data, it is already a very capable model.

When using a pretrained model, <code>cnn_learner</code> will remove the last layer, since that is always specifically customized to the original task and replace it with one ore more layers with randomized weights, of an appropriate size for the dataset we are working with. This last part is known as the head of the network.

* **Transfer learning**.: using a pretrained model for a task different from it was originally trained for.
* **Fine tuning**: A transfer learning technique that updates the parameters of a pretrained model by training for additional epochs using a different task that the one used for pretraining.

In our example, we called:

```
learn = cnn_learn(dls, restnet34, metrics = error_rate)
learn.fine_tune(1)
```

When using <code>fine_tune(1)</code>, we are asking fastai to use one epoch to fit just those parts of the model necessary to set the new random head to work correctly with the new dataset. <code>fine_tune</code> considers two kinds of epochs, those from the freeze run and from the proper run. When freezing, it runs an epoch without modifying the weights of the pretrained model. This way, it can learn the head of the network using a good starting point. Then, it unfreezes all the parameters (wights) and trains the whole model.

### Smart definition of test sets
To do a good job at defining a validation set (and possibly a test set), we will sometimes want to do more than simply grabbing a random fraction of the original dataset.

**Remember:** a key property of the validation and test sets is that they must be representative of the new data you will see in the future. While this may seem impossible to achieve, we can accomplish similar results.

For example, in time-series data, instead of considering random points of data for the validation and/or test sets, we can use periods of time in the future that have not been seen by the model.

We could use cross-validation in this environment, but it would be tricky. Another option would be to use multiple validations sets. For cross validation, the idea would be to divide the time-series in slices and consider them in order. That way:
* first fold trains using slice (1) and evaluates with (2).
* second fold trains using (1,2) and evaluates with (3).
* third fold trains using (1,2,3) and evaluates with (4).
* etc.

Another example occurs in computer vision problems, when we can easily anticipate that the learned model will be used with data that may be qualitatively different from the data used during training.

If we want to predict if a driver is distracted, it would be appropriate to hold pictures from drivers that we not consider during training (i.e., not use all of the people from the study). This way we would consider a group of people during training and validation, and another different group of people in testing. Of course, each pearson usually has multiple associated images.