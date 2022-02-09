## 2 - FROM MODEL TO PRODUCTION

### The drivetrain approach
The basic idea is to start considering your objective and what you want to achieve. Then thi about what actions need to be done by you to meet the objective and what data you have (and can adquire) that would help. Finally, build a model that you can use to determine the best actions to take in order to get the best results in terms of your objective.

Therefore, a crucial aspect of building machine learning models (especially predictive ones) is to consider **what data we already have and what data we will additionally need to collect*. Both will determine which models we can build.

As an example, consider a recommendation engine. The **objective** of this system is to drive additional sales by offering good recommendations to the user of the application. These recommendations would offer items the user would not be aware of without the engine. In this case, the **action** is to rank recommendations to evaluate their quality and ability to boost sales. This will require "new data" that will result from conucting experiments on a wider range of recommendations and customers. Then, you could build two models for purchase probabilities, conditional on seeing or not seeing a recommendation. The difference between these probabilities is a **utility function for a given recommendation** to a customer. It will be low in those cases where the algorithm recommends a familiar item to customer that has already knew the item and already rejected it (i.e., both probabilities are low) or when the item would be bough even without the recommendation (i.e., both probabilities are high ad cancel each other in the utility function).

