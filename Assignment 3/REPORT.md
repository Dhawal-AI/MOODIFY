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

![ML Validation set](https://user-images.githubusercontent.com/81472530/114302049-f3e4a100-9ae4-11eb-85bd-9df81d6736f1.jpg)

This module emphasises on creating an additional data set called validation set to validate the training data and further tune the hyperparameters.
Finally using the model well built on training and validation data to make predictions on the test data set.
This is a better workflow because it creates fewer exposures to the test set.
The programming exercise focuses on shuffling and dividing the single data set and checking the effectiveness of this workflow.




