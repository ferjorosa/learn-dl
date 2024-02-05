## 9 - TABULAR MODELING DEEP DIVE

### Entity embeddings of categorical variables

The task of entity embedding is to map categorical values to a multi-dimensional space where values with similar function output are close to each other.

Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables. It is especially useful for datasets with lots of high cardinality features where other methods tend to overfit. As entity embedding defines a distance measure for categorical variables, it can be used for visualizing categorical data and for data clustering.

The original paper of <a href="https://arxiv.org/pdf/1604.06737.pdf">Guo and Berkhahn (2016)</a> proposes a neural network architecture that introduces those embeddings along continuous features. That way, the embeddings are learned considering other important information of the dataset. 

We could theoretically use other dimensionality reduction techniques (e.g., PCA, TSNE, etc.). They have a set of advantages and disadvantages:
* Good: separated from the model, they could be used with multiple different approaches
* Bad: could return considerably worse results, for example PCA with one-hot encoding <a href="https://stats.stackexchange.com/questions/159705/would-pca-work-for-boolean-binary-data-types">is not usually recommended</a>. Another possibility is to use multiplie correspondence analysis (MCA).

### Beyond deep learning

Although deep learning has show to be clearly superior for unstructured data, recent studies have shown that ensembles of decision trees (i.e., bagging and boosting) show similar results to neural networks for many kinds of structured data. The advantage of ensembles resides in their ability to:
* Train faster.
* Being  easier to interpret.
* Not requiring special GPU hardware for inference at scale.
* Often requiring less hyperparameter tuning.

Most importantly, the critical step of interpreting a model of tabular data is significantly easier for decision tree ensembles.  There are tools and methods for answering the pertinent questions, like:

* Which columns in the dataset (i.e., features) were the most important in our predictions?
* How are these features related to the target variable?
* How do they impact with each other?
* Which particular features were most important for some particular observation?

### Data preparation: handling dates

An important data preparation step in most datasets is to enrich our representation of dates. To help our algorithm handle dates more intelligently, **we would like our model to know more than wether a date s more or less recent than another**. 

We might want our model to to make decisions based on that date's day of the week, on wether a day is a holiday, on what month it is in, and so forth. To do this, we replace every date column with a set of date metadata columns, such as holiday, day of the week, and month.

Fastai comes with a function that will do this for us, we just have to pass a column name that contains dates `add_datepart`:

```
df = add_datepart(df, saledate)

# variables añadidas:
saleYear saleMonth saleWeek saleDay saleDayofweek saleDayofyear
> saleIs_month_end saleIs_month_start saleIs_quarter_end saleIs_quarter_start
> saleIs_year_end saleIs_year_start saleElapsed
```

### Test and validation with temporal data

If your data is a time series, choosing a random subset of the data will be both too easy (you can look at the data both before and after the dates your are trying to predict) and not representative of most business use cases (where you are using historical data to build a model for use in the future). If your data includes the date and you are building a model to use in the future, you will want to choose a continuous section with the latest dates as your validation set (for instance, the last two weeks or last month of the available data).

With time series data, it is usually good practice that the validation set  is in later time than the training set. In addition, it would be appropriate to have the test set even later in time.

**Example:**

* Test set 05/2012 - 11/2012
* Validation set 11/2011 - 04/2012
* Trainig set before 11/2011

<a href="https://www.fast.ai/2017/11/13/validation-sets/">FastAI blog post</a>

### Decision trees and categorical variables

According to the FastAI book, random forests can inherently work with categorical variables (integer form I assume). However, I dont see the appeal when the number of values is high. In that case, the interpretability may be lost due to the use of integer representation if the variable is not ordinal.

My opinion is to use feature engineering to extract categorical variables with less values (I would not use modelID if it has 3000 values for example).

It is a topic that I am still pondering about.

### Random forests

In essence, a random forest is a model that averages the predictions of a large number of decision trees, which are generated by randomly varying various parameters that specify what data is used to train the tree and other tree parameters. Random forests are based on the **bagging** approach, which focuses on minimizing variance of predictions by considering deep trees with different subsets of features and data instances, and then average their respective predictions.

#### Out-of-bag error

The out-of-bag error is a way of measuring prediction error in the training data by including in the calculation those training rows that were not considered by the specific tree.

The intuition is that since every tree was trained with a different randomly selected subset of rows, out-of-bag error is a little like imagining that every tree has its own "validation set". That validation set is simply the rows that were not selected for that tree's training.

It is particularly beneficial in cases where we only have a small amount of training data, as it allows us to see whether our model generalizes without removing items to create a validation set.

#### RF interpretation: tree variance for prediction confidence

The random forest averages the individual tree's prediction to get an overall prediction. But how can we know the confidence of the estimate?

One simple way is to use the standard deviation of predictions across the trees, instead of just the mean. This tells us the relative confidence of predictions. 

**Example:**

```
preds = np.stack([t.predict(valid_df) for t in m.estimators_])
preds.shape
(40, 7988) # 40 trees, 7988 validation rows
```

This information is very useful in a production setting, and could be further considered.

#### RF interpretation: feature importance

Feature importance give us an idea of how our model is making its predictions. The way these importances are calculated is quite simple yet elegant.

The feature importance algorithm loops through each tree, and then recursively explores each branch. at each branch, it looks to see what features  was used for that split and how much the model improves as a result of that split. The improvement (weighted by the number of rows in that group) is added to the importance score for that feature. This is summed across all branches of all trees, and finally the scores are normalized such that they add to 1.

#### RF interpretation: partial dependence

Partial dependence plots try to answer the question: if a row varied on nothing other than the feature in question, how would it impact the dependent variable?

To answer it, lets consider an example where we have a "Year" variable with 50 categorical values. This variable represents the year of sale and goes from 1950 to 2000. In order to estimate the partial dependence of "Year", we replace every single value in the dataset 1950 and then calculate the predict sale price (targe variable) for every sale, taking the average over all data instances. Then we do the same for 1951, 1952, and so forth until our final year of 2000. 

This isolates the effect of "Year" and (even if it does so by averaging over some imagined data instances where we assign a "Year" value that might never actually exist alongside some other values).

With these averages we can then plot each year on the x-axis, and each prediction on the y-axis. This, finally, is a partial dependence plot

#### Extrapolation with RFs

Since a random forest just averages the predictions of a number of trees, and a tree simply predicts the average value of the rows in a leaf, then a random forest can mever predict values outside the range of the training data. This is particularly problematic for data indicating a tren over time, such as inflation. In those cases, our prediction will be systematically too low.

For time variables, this can be approached by considering a linear regression model in leaf nodes instead of the average value.

### Data leakage

Data leakage is usually is usually found when the information of the target variable is "leaked" into other features. For example, when the value of a feature was written after we knew the value of the target variable.

It is usually found in apparently meaningless data columns that have great feature importance, also combined with a model accuracy that is "too good to be true".

A way to identify this problem is to analyze these columns via partial dependence plots, to see if particular values of those columns could be leaked from the target variable.

### Combining embeddings with other methods

As a summary, the embeddings obtained from the trained neural network can boost the performance of machine learning methods considerably when used as the inpute features instead.
