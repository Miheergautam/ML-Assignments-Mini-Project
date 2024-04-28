Mini project Solutions

1. Plot the waveform for data from each activity class. Are you able to see any differences/similarities between the activities? You can plot a subplot having six columns to show differences/similarities between the activities. Do you think the model will be able to classify the activities based on the data?![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.001.png)

Solution:

- From the waveforms of the activities, our team could identify the following similarities and differences between the activities:
  - The variance of stationary activities (sitting, standing, laying) is way less than that of dynamic activities (walking, walking upstairs, walking downstairs)
  - To differentiate between laying and other activities, one can look at the fact that the value of acceleration in axis 1 is less than that of others. 
  - To differentiate between sitting and standing, we can see that the acceleration of axis 2 is greater than that of axis 3 and vice versa.
  - It is difficult to differentiate between dynamic activities, but walking upstairs has a negative acceleration in axes 2 and 3.  
- Yes, our team believes that the model should be able to confidently, with reasonable accuracy, differentiate between static and dynamic activities. It should be able to differentiate between all the three different static activities. We expect that it will face challenges in differentiating between dynamic activities. 



1. Do you think we need a machine learning model to differentiate between static activities (laying, sitting, standing) and dynamic activities(walking, walking\_downstairs, walking\_upstairs)? Look at the linear acceleration for each activity and justify your answer

Solution:

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.002.png)

- Looking at the plots of total acceleration, we can see that the variance in the dynamic activities is way higher than that of static activities. Thus, by just using the acceleration in three axes and total acceleration, we might be able to differentiate between static and dynamic activities. 
- Using the acceleration in three axes and total acceleration, we might be able to differentiate between static activities, but using machine learning would give a higher accuracy.
- However, differentiating between the different dynamic activities like walking, walking upstairs, and walking downstairs would not be possible with the use of strong machine learning algorithms. 

1. Train Decision Tree using trainset and report Accuracy and confusion matrix using testset.

   Solution:

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.003.png)

1. Train Decision Tree with varrying depths (2-8) using trainset and report accuracy and confusion matrix using Test set. Does the accuracy changes when the depth is increased? Plot the accuracies and reason why such a result has been obtained.

   Solution 

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.004.png)

- As we can interpret from the plots, as we gradually increase the depth of the decision trees, the accuracy initially increases, then reaches a maximum, and then decreases slightly and becomes almost constant. 
- This behavior highlights the trade-off between bias and variance. Initially, at the lower depths, the model is not complex enough. 
- However, at higher depths, we are no longer able to increase accuracy; rather, we have a lower accuracy because the model is trying to include outliers in the training data set. 
1. Use PCA (Principal Component Analysis) on Total Acceleration to compress the acceleration timeseries into two features and plot a scatter plot to visualize different class of activities. Next, use TSFEL (a featurizer library) to create features (your choice which ones you feel are useful) and then perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities. Are you able to see any difference?

Solution: Principal Component Analysis (PCA) is a technique used for dimensionality reduction, which can be particularly useful for visualizing data in lower-dimensional space. Here's a theoretical explanation for the steps:

Principal Component Analysis (PCA) is a technique to simplify and visualize data. In the first scenario, we compressed total acceleration data into two components and plotted them, aiming to see if different activities form distinct groups. In the second scenario, we used a feature extraction library called TSFEL, specifically the mean\_abs\_deviation function, to create features. We then applied PCA to these features and visualized the results in a scatter plot. However, despite these efforts, we observed that the activities were not clearly separated in both cases. This suggests that the chosen features may not be sufficient for distinguishing between activities, and further exploration of different features might be necessary.

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.005.png)

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.006.png)

1. Use the features obtained from TSFEL and train a Decision Tree. Report the accuracy and confusion matrix using test set. Does featurizing works better than using the raw data? Train Decision Tree with varrying depths (2-8) and compare the accuracies obtained in Q4 with the accuracies obtained using featured trainset. Plot the accuracies obtained in Q4 against the accuracies obtained in this question.

   Solution:

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.007.png)

![](Aspose.Words.b2ccc880-1398-469d-a8a4-c59a91d8c4a1.008.png)

1. Are there any participants/ activitivies where the Model performace is bad? If Yes, Why? 

   Solution: In our analysis, we identified specific instances where the model's performance exhibited shortcomings, particularly in distinguishing between dynamic activities. Notably, the confusion matrix revealed notable challenges in classifying dynamic activities accurately. Two specific dynamic activities, namely "WALKING\_DOWNSTAIRS" and "WALKING," emerged as particularly problematic for the model.

The underlying reason for this suboptimal performance can be attributed to the intrinsic similarity in the acceleration patterns associated with "WALKING\_DOWNSTAIRS" and "WALKING" activities. These dynamic activities inherently share commonalities in their acceleration values, making it difficult for the model to discern subtle differences solely based on this feature. Since our model relies exclusively on acceleration values as features, the overlapping nature of these activities in the feature space contributes to misclassifications.

In essence, the model faces challenges when tasked with distinguishing activities that exhibit similar acceleration characteristics. The limitations arise from the inherent complexities and subtle variations in the acceleration patterns of certain dynamic activities. Addressing this issue may require additional features or more sophisticated modeling techniques that can capture nuanced distinctions within the data, thereby enhancing the model's ability to discriminate between closely related activities.

It's crucial to note that the model's effectiveness may hinge on the quality and representativeness of the dataset, as well as the chosen features and model parameters.




