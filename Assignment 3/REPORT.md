# REPORT:
## Framing-
ML systems learn how to combine input to produce useful predictions on never-before-seen data.

### 1. Training
Training means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.
### 2. Inference
Inference means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions.
### Regression VS Classification
Regression based algorithms as used mostly for continuos data sets,where we predicts housing prices and Classification based models basicary work in binary(more or less) that is predicting whether a tumor is malignant or benign or an email is spam or not-spam



![image](https://user-images.githubusercontent.com/81459933/114316069-86a23180-9b1f-11eb-9e0c-7d0e03efd062.png)
![image](https://user-images.githubusercontent.com/81459933/114316113-c49f5580-9b1f-11eb-9549-32079ac35cc2.png)

## Descending into ML-
Using the equation for a line, you could write down this relationship as follows:
y=mx+c and this is basically what we use in our linear regression models hence suggesting the word linear.
Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight (
w1,w2 
, etc.). For example, a model that relies on three features might look as follows:
y=c+w1x1+w2x2
Training a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called empirical risk minimization.

Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater.
The loss function y2 is (y-y')^2
![image](https://user-images.githubusercontent.com/81459933/114316933-5a88af80-9b23-11eb-9642-5fcdd2804fef.png)

## Reducing loss
Gradient descent mechanism is an effective way to minimise the error in lesser number of iterations. A random initial value is picked and the next weight is chosen in the direction of the negative gradient to approach the converging point faster. Gradient descent algorithms multiply the gradient by the hyperparameter: learning rate to optimise the iterations taken to achieve the minimum loss. The conclusion from the playground exercise was that a smaller learning rate takes much longer time to converge and a large learning rate may not converge at all(climbing the curve instead of descending to the bottom). 

![image](https://user-images.githubusercontent.com/81459933/114316933-5a88af80-9b23-11eb-9642-5fcdd2804fef.png)

![image](https://user-images.githubusercontent.com/81459933/114316959-755b2400-9b23-11eb-887c-ec939086ff26.png)
### Learning Rate
There's a Goldilocks learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.
if rate is too small,it leads to many iterations and usage of memory whereas a larger learning rate means overshooting

![image](https://user-images.githubusercontent.com/81459933/114317005-ac313a00-9b23-11eb-87ba-ea70f549a267.png)

![image](https://user-images.githubusercontent.com/81459933/114317026-c539eb00-9b23-11eb-9ae4-b03ecfe2fb1e.png)
Mini-batch stochastic gradient descent (mini-batch SGD) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

![image](https://user-images.githubusercontent.com/81459933/114317041-d4209d80-9b23-11eb-9240-f2562f5b7027.png)

## Generalisation:
Basically since polynomials are infinite you can always exactly or perfectly fit a polynomial to your data-set. the problem with that is it results in loss of generalisation, which means 99% of your predictions will be bogus.
An overfit model gets a low loss during training but does a poor job predicting new data. If a model fits the current sample well, how can we trust that it will make good predictions on new data? As you'll see later on, overfitting is caused by making a model more complex than necessary. The fundamental tension of machine learning is between fitting our data well, but also fitting the data as simply as possible
The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.
## Training and Test Sets
The previous module introduced the idea of dividing your data set into two subsets:

#### training set—a subset to train a model.
#### test set—a subset to test the trained model.

Never train on test data. If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. For example, high accuracy might indicate that test data has leaked into the training set.

## Validation set

Assessment of check your intuition:

<img width="563" alt="Screenshot 2021-04-18 at 1 00 53 AM" src="https://user-images.githubusercontent.com/74460296/115124753-90290f00-9fe1-11eb-9610-71e316d26a13.png">


This module emphasises on creating an additional data set called validation set to validate the training data and further tune the hyperparameters.
Finally using the model well built on training and validation data to make predictions on the test data set.
This is a better workflow because it creates fewer exposures to the test set.
The programming exercise focuses on shuffling and dividing the single data set and checking the effectiveness of this workflow.

## Representation
The basic qualities that a good feature should possess are :
It should have a clear meaning .
It should occur quite a few times , features such as unique ID etc are of no use.
We should check that the feature isn’t noisy
We shouldn’t mix magic values with the feature if a particular value is missing . Instead we can create a separate boolean feature to indicate whether or not a value was supplied.

## Feature Crosses:
A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together. However feature crosses between categorical features are sometimes more useful than numerical features. Thus we cross the one hot encodings of these features. From these crosses we get  binary features that can be interpreted as logical conjunctions. From such feature crosses we get more predictive ability than either feature of their own.
<img width="731" alt="Screenshot 2021-04-18 at 1 06 53 AM" src="https://user-images.githubusercontent.com/74460296/115124923-b26f5c80-9fe2-11eb-9a2b-00fd083f823f.png">
Certain non-linearity in the model can be encoded by crossing two or more existing features to produce a synthetic feature(feature cross). The new feature can also be added to the linear formula just like any other feature. 

## Regularisation:
 As we can see that by simply removing the feature crosses we are able to get a much simpler linear model rather than the initial complicated curve.  In order to overcome this problem we do regularization. We try and keep the model as simple as possible. Instead of just doing empirical risk minimization we now start using structural risk minimization. We try to minimize both loss and the complexity. In this topic we analyze model complexity as a function of the weights of all the features in the model. We quantify complexity using the L2 regularization formula, which defines the regularization term as the sum of the squares of all the feature weights.
 ![Screenshot 2021-04-18 at 1 07 30 AM](https://user-images.githubusercontent.com/74460296/115124990-0e39e580-9fe3-11eb-89de-9a980307d3d1.png)
<img width="734" alt="Screenshot 2021-04-18 at 1 07 42 AM" src="https://user-images.githubusercontent.com/74460296/115125000-1560f380-9fe3-11eb-835c-4df72d0589ab.png">

 
Lambda is the factor accounting for the amount of regularisation effect. Increasing lambda makes the weight distribution more like the Gaussian bell curve and lowering its value results in flatter distribution.

## Logistic Regression
This is basically used to solve discrete valued problems working in binary of 0 and 1. it can include things like whether a tumor is malignant or benign
Regularization is also very important in logistic regression. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. If you don't specify a regularization function, the model will become completely overfit.

## Classification
In order to map a logistic regression value to a binary category , we must define a classification threshold . In the example of spam and not spam emails, a value above the threshold would be mapped to spam . This threshold value can’t be set to 0.5 always and we need to consider a variety of metrics before deciding our threshold.
Some terms that we need to know are :
A true positive is an outcome where the model correctly predicts the positive class. Similarly, a true negative is an outcome where the model correctly predicts the negative class.
A false positive is an outcome where the model incorrectly predicts the positive class. And a false negative is an outcome where the model incorrectly predicts the negative class.
Based on this , we define some of our metrics.
We can summarize the model using a 2x2 matrix containg the four possible outcomes namely true positive(TP), false positive(FP), true negative(TN), false negative(FN). Then evaluate classification models using metrics(accuracy, precision and recall) derived from these four outcomes. Accuracy is the fraction of correct predictions. Accuracy fails to do a good job in class-imbalanced sets. Precision is the fraction of correct positive predictions.
We first categorize this column of our training data according to our threshold. After this we build and train the logistic regression model on our training dataset. We plot accuracy, precision and recall for this model. We can then experiment with classification threshold to see which one produces the highest accuracy. We see that a threshold of 0.5 causes highest accuracy.
![Screenshot 2021-04-18 at 1 08 07 AM](https://user-images.githubusercontent.com/74460296/115125206-32e28d00-9fe4-11eb-8810-f28606bf497c.png)
<img width="750" alt="Screenshot 2021-04-18 at 1 08 34 AM" src="https://user-images.githubusercontent.com/74460296/115125213-39710480-9fe4-11eb-8b57-0dc27fc6fb16.png">
![Screenshot 2021-04-18 at 1 08 41 AM](https://user-images.githubusercontent.com/74460296/115125234-4e4d9800-9fe4-11eb-88e3-5e999698a878.png)




