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




