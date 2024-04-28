
Question 6

If we look at the overall performance of all three models, we see that the random forest classifier performs slights better than the dicision tree classifier. Still, the linear regression model, which has been used to classify, performs poorly. 

This is expected as using Linear regression here is not justified  because:
Linear regression model assumes that a linear relationship exists between the variables, which is a very specific assumption. The task we have chosen is also a classification task. Thus, such a relationship does not exist here. 
Linear regression model is a comparatively simple model, whereas decision trees and random forests are more complex and would be able to capture more complex and non-linear relationships. 
Linear regression model is more prone to overfitting as it would give more weightage to outliers, but the random forest and decision tree do not have this problem. 
Linear regression model does not have the ability to perform the selection of features inherently, while the decision tree and the random forest classifiers do, and thus, they perform better. 
Additionally, regressors are not used for classification tasks as they are not designed for classification. Thus, using the linear regression model is not justified. 

For the Decision Tree Classifier, 
Accuracy = 76.69%

For Random Forest Classifier, 
Accuracy = 81.81%

For Linear Regression, 
Accuracy = 65.15%
MSE = 0.262

Question 7 

The top 10 Important Features of the Linear Regression model are:
[ 89  90  94  93  95  97  96  99  92 100]

The top 10 Important Features of the Random Forest model are:
[101   3   0 119 120 107  80 114  39 121]
The sum of Feature Importances from Random Forest: 1.0000000000000002

We infer that:
None of the important features of the random forest and linear regression models match. This means that both models work very differently. 
Linear regression weights are restricted to a very small range of 89-100, whereas the features range over 0 to 125 (approx). Thai highlights that the random forest model performs the feature selection better, which might be the reason for its better performance and higher accuracy. 

