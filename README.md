# Airbnb Price Prediction
### Charlie Hipps

## Introduction

This goal of this project was to research the best ways predict the price category (scale 0 to 5) of AirBnb listings in the Los Angeles area based on scraped data about each listing as part of a Kaggle competition. A full description of the competition and datasets can be found here: https://www.kaggle.com/competitions/duke-cs671-fall23-airbnb-competition/overview. Along with accurate price prediction, the goal of this project was to compare the performance of various machine learning networks and prediction techniques. Below is a complete summary of my work.

## Exploratory Analysis

The first step in approaching this problem is analyzing the given dataset. The dataset has 46 features other than price, but upon initial observation, it was clear that some would be much more useful than others. For example, features information on the scraping of the data (`scrape\_id`, `last\_scraped`, and `calendar\_last\_scraped`) will likely not be as useful as location data (`latitude` and `longitude`). Eliminating extraneous features will simplify the models while maintaining accuracy, and possible reducing overfitting to the given dataset. However, before making assumptions about the relevancy of certain features, we must do a more formal analysis of the dataset.\\
\\
To help our understanding of the features throughout analysis, I will partition the features into four categories: metadata, booking information, property description, and property location. Here is a table of the categorization of the features (some are abbreviated and/or grouped):\\
\\
[Categorization of Features](/visualizations/outputs/feature_categories.png)

We will use these categories to guide our analysis as we visualize the dataset. I expect that data within the same category will often be closely related.

To begin, we will look at various visualizations of the dataset. These will all focus on the relationship between various features and price. First, let's examine the property description category. Notice the similar patterns between the `beds` and `accommodates` features shown Figures \ref{fig:beds_hist` and \ref{fig:accom_hist`. Both have a similar effect on the price distribution, but `accommodates` seems to have a more consistent effect, while also not having empty values like `beds` does. `beds` and `accommodates` are also highly correlated, so it may prove useful to only use the `accommodates` feature instead of both.\\
\\
Now, consider the booking information category. There are less clear relationships between price distributions and the variables in this category. This is shown in the figures below, which show the changes in distribution caused by `number\_reviews\_ltm` and `availability\_365`, respectively. However, there are slight differences in the distributions, so there may be value in including these in models.\\
\\
[Price versus Review History bar chart](/visualizations/outputs/reviews_ltm_hist.png)\\
\\
[Price versus Availability History bar chart](/visualizations/outputs/avail_365_hist.png)\\
\\
The last category I looked at visualizations for was the location category. The map below was produced using the relationship between `latitude` and `longitude` data and price. There are no clear boundaries between prices based solely on location, but it is clear that there are higher and lower priced areas. There are more mapping visualizations that I did, while separating based on property description features that are shown in the submitted code, but there are always outliers within the map.\\
\\
[Map of locations color-labeled by price category](/visualizations/outputs/lat_long_full.png)\\
\\
Beyond these visualizations, there are some other important things to note about the dataset. Most importantly, there are many features that are very irrelevant to the price. Most notably, metadata like `scrape\_id` and `last\_scraped` need to be dropped. `last\_scraped` only has two unique values and seems to be arbitrary, while `scrape\_id` is unique for all listings and would only lead to overfitting. In addition, features like `picture\_url` and `description` require unique processing, so they will be difficult to include. I may experiment with these, but an initial exploration of the images was not promising. They all seem to be of random parts of the properties and some of the images cannot be reached.\\
\\
The summary of this analysis is it appears the models can be simplified by removing some correlated features, but the lack of clear relationships mean that it may be useful to use as many processable, relevant features as possible.

## Models

I experimented with three primary types of models for this classification task. The motivations for each model are discussed below.

### Neural Networks

The first type of model I tried to apply was a neural network. I used this first because of its ease of use and my personable familiarity with it. In addition to their simplicity, they are relatively quick to train with the size of data we are given, and there are very good public libraries for building neural networks.

### Random Forests

The next type of classifier I tried was a random forest. I picked this for many of the same reasons as neural networks, except they take slightly longer to train, especially when many the number of estimators used is large. This was my next choice because some of the data visualization work I did suggests that tree based classifiers could have success. Although the data is too complicated for a reasonable single decision tree, random forests seemed like a good way to leverage this.

### Boosting

Finally, I wanted to explore the viability of using a boosting algorithm on the dataset. Since we experimented with various boosting libraries during homeworks for this course, I knew there were good ones available. I ended up deciding on XGBoost because I felt the most comfortable with it.

## Training

### Neural Networks

The primary neural networks design I used was a the Sequential network 
(a feedforward neural network) from Keras. Note I setup the model to for multiclass classification. The price is converted into a one-hot encoding with six categories, and the final output layer uses the softmax function to indicate the most likely category (price) for the input data. During training, the weights of the network are updated iteratively to minimize the categorical crossentropy loss. This is done with backpropagation and stochastic gradient descent according to the optimizer function (adam was used in this case).

The training in this case was always very quick. It only required a few seconds per epoch for the standard neural networks.

Special note: I experimented with incorporating the property description and image into a neural network. I first tokenized the descriptions and fed them to an Embedding layer. I also used a pre-trained MobileNetV2 model for the images. This training took much longer but yielded very poor accuracy, so I dropped this approach.

### Random Forests

The training process for a random forest classifier involves creating an ensemble of decision trees. Initially, the dataset is divided into training and validation sets. For each tree in the ensemble, a subset of the training data is sampled with replacement. Each decision tree is then constructed following the standard decision tree building process, where, at each node, the algorithm selects the best feature among a random subset of features. This randomness helps decorrelate the trees and reduces overfitting. The trees are grown until a specified stopping criterion is met, often defined by the maximum depth of the tree or the minimum number of samples required to split a node. In this case, the predictions from individual trees are aggregated using a majority vote to classify each point after training is complete.

Training the final decision forest with 500 estimators took about 20 to 30 seconds. This only created a problem during the parameter selection process, which is discussed in the section below. Because an individual forest would take that long, I chose to break the grid search for parameters into two parts.

### Boosting

The training process for an XGBoost classifier also involves a collection of decision trees but differs from random forests in the optimization strategy and regularization techniques. XGBoost builds trees sequentially, with each tree aiming to correct the errors of the previous ones. It uses a gradient boosting framework, optimizing a loss function by minimizing the gradient of the loss with respect to the model's predictions.

During the training process, XGBoost incorporates regularization techniques to prevent overfitting. Regularization terms, such as L1 and L2 regularization on the weights, are added to the objective function. Additionally, the learning rate controls the contribution of each tree to the final prediction, and early stopping is often employed to halt training when performance on a validation set ceases to improve.

## Hyperparamter Selection

### Neural Networks

For the neural networks, the number of layers and amount of neurons per layer were trained largely through trial and error. I never got to a high enough accuracy to warrant a detailed tuning of the structure.

## Random Forests

To select optimal parameters for the random forest classifier, I did two grid searches with cross validation. The first one was for the number of estimators in the forest. I found higher the better, but that the performance was approaching optimal around 150 estimators. This allowed me to used 150 estimators as I did a second grid search over some of the other parameters. Splitting the search allowed me to save time in the parameter selection. For the final model, I used the parameters found in the second search with more estimators (500).

A graph of the effect of the number of estimators used in the forest on the accuracy can be seen in the figure below. This was produced by the first round of grid search.

[Plot of Number of Estimators versus Accuracy](/models/graphics/rf_ests_vs_acc.png)

## Boosting

Similar to the random forest classifier, I used a grid search with cross validation across three parameters for XGBoost. A graph of the effect of learning rate used in the training of the model on the accuracy can be seen in the figure below. This was produced by the first round of grid search.

[Plot of Learning Rate versus Accuracy](/models/graphics/learn_acc.png)

## Data Splits

Other than for the neural network classifiers, which were more experimental, I always used 5-fold cross validation to split the data for training and validating. This helped reduce overfitting of the models in testing. This was done using the GridSearchCV method.

## Errors and Mistakes

The primary mistake I made in this competition was spending too much time exploring approaches that didn't end up working. This first happened in my experimentation with neural networks on this dataset. I tried many different structures and tried many different subsets of features hoping to improve performance. I should have known earlier in the process that it was more worth moving on to a different approach because there was no way these tweaks could have overcome the gap that I eventually found to other types of classifiers.

A second mistake I made in the same vein was spending too much time attempting to incorporate the property images and descriptions into the classification. I have experience from a past project with language processing, so I thought I might be able to gain some insights in this area. Although this work was interesting, I should have stuck with my intuition from the data exploration stage that the images and descriptions were not consistent enough to add real value to the classification. 

## Predictive Accuracy

I will discuss the predictive accuracy of each type of classifier separately.

### Neural Networks

The best training accuracy I could get form the neural networks was always between about 40 and 42 percent. I never submitted a prediction to Kaggle, so the acurracy on the test set is unknown.

### Random Forests

Even with poor parameter selection, random forests clearly had better accuracy than the neural networks. As previously mentioned, I used 5-fold cross validation to find parameters. The worst parameter groups in this process had validation accuracy around 49 percent, while the best had accuracy close to 55 percent.

The best submission to Kaggle, had an accuracy of 53.86 percent on the test dataset.

### Boosting

Similar to random forest classifiers, the boosting classifiers clearly had better performance than the neural networks. However, for some of the poorer combinations of parameters, the validation accuracy was around 43 percent, while the best was nearly 53 percent.

The best submission to Kaggle had an accuracy of 52.9 percent on the test dataset.