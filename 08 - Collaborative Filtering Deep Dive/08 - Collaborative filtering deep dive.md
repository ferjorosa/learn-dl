## 8 - COLLABORATIVE FILTERING DEEP DIVE

One common problem is having a number of users and a number of products, and you want to recommend which products are most likely to be useful for which users.

A general solution to this problem is called **collaborative filtering** and it works like this: 
1. look at which products the current user has used or liked.
2. find other users who have used or liked similar products.
3. recommend other products that those users have used or liked.

### MovieLens data

We can represent the matchings between each user and each movie as a giant matrix. The empty cells in that matrix would be the things our model should learn to fill in. Those are the places where a user has not reviewed the movie yet, presumably because they have not watched it. For each user,we would like to know which of those movies they might be more likely to enjoy.

If we knew for each user to what degree they liked each important category that a movie might fall into, such as genre, age, preferred riectors and actors, and so forth, and we knew the same information about each movie, then a simple way to fill in this table would be to multiply this information together for each movie and use a combination.

For instance, if we consider 3 aspects such as "science fiction degree", "novelty degree", and "action degree" where each of these aspects lies in the [-1, 1] range, we could represent movies and user likes in the following way:

```
star_wars: [0.98, -0.9, 0.9]
casablanca: [-0.99, -0.97, 0.4]
user_1: [0.9, -0.6, 0.8]


(user_1 * star_wars).sum = 2.14
(user_1 * casablanca).sum = 0.01
```

We can see that user 1 likes old movies from the science fiction genre with much action. We can check for a match by multiplying and summing. By doing so, we see that Star wars may be a good recommendation, but Casablanca may not be a good one.

It is generally impossible to know beforehand the true means of our representation. That is why they are known as **latent factors**. Since we don't know what the latent factors are, and we don't know how to score them for each user and movie, we should **learn them**.

### Learning the latent factors
The idea for our movie ratings example is to randomly initialize some parameters (i.e., the set of latent values for each user and movie). Then, optimize these parameters using **stochastic gradient descent** with a loss function such as the **mean square error**.

At each step, the optimization process will calculate the match between each movie and each user using the dot product and will compare it to the actual rating that the user gave to each movie. I will then calculate the derivative of this value and step the weights by multiplying this by the learning rate. This will iteratively improve the loss and the recommendations.

