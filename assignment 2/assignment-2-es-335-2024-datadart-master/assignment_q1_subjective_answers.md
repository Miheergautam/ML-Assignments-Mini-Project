Question 1: 

(i) Implement full-batch and stochastic gradient descent. Find the average number of steps it takes to converge to an epsilon-neighborhood of the minimizer for both datasets.
    Visualize the convergence process for 15 epochs. Choose epsilon = 0.001 for convergence criteria. Which dataset and optimizer takes a larger number of epochs to converge, and why?
    Show the contour plots for different epochs (or show an animation/GIF) for visualisation of optimisation process. Also, make a plot for Loss v/s epochs.

# Dataset 1:

          num_samples = 40
          np.random.seed(45)  
          
          # Generate data
          x1 = np.random.uniform(-20, 20, num_samples)
          f_x = 100*x1 + 1
          eps = np.random.randn(num_samples)
          y = f_x + eps

# Dataset 2:

          np.random.seed(45)
          num_samples = 40
              
          # Generate data
          x1 = np.random.uniform(-1, 1, num_samples)
          f_x = 3*x1 + 4
          eps = np.random.randn(num_samples)
          y = f_x + eps

# Taken:

      number of epochs = 1000
      epsilon = 0.001
      learning_rate = 0.0001

# Observation:

(A) Find the average number of steps it takes to converge to an epsilon-neighborhood of the minimizer for both datasets.

    Number of epochs required to converge for Dataset 1 using full-batch gradient descent:  18
    Number of epochs required to converge for Dataset 1 using stochastic gradient descent:  22349

    Number of epochs required to converge for Dataset 2 using full-batch gradient descent:  140
    Number of epochs required to converge for Dataset 2 using stochastic gradient descent:  20803

(B) Which dataset and optimizer takes a larger number of epochs to converge, and why?

# Dataset 1: 

* Full-Batch Gradient Descent: Converged in 18 epochs. Since Dataset 1 is relatively small (40 samples), full-batch gradient descent efficiently processes the entire dataset in each epoch, quickly reaching convergence.  

* Stochastic Gradient Descent (SGD): Required 22349 epochs to converge. SGD updates parameters using only one sample at a time, leading to more frequent updates but with higher variance. This high variance slows down convergence, requiring significantly more epochs compared to full-batch gradient descent.

# Dataset 2:

* Full-Batch Gradient Descent: Converged in 140 epochs. Dataset 2, like Dataset 1, is relatively small (40 samples), allowing full-batch gradient descent to efficiently process the entire dataset in each epoch and converge quickly.

* Stochastic Gradient Descent (SGD): Required 20803 epochs to converge. Similar to Dataset 1, SGD struggles with convergence due to the higher variance introduced by processing one sample at a time. Despite the small dataset size, the noise added to the data (especially evident in Dataset 2) complicates the convergence process for SGD.

# Overall:

* Full-batch gradient descent typically converges faster than SGD on smaller datasets, as it processes the entire dataset in each epoch.

* SGD, while more computationally efficient per iteration, can struggle with convergence on datasets with added noise or when the dataset size is small, as it requires more epochs to approximate the minimum due to its stochastic nature.

(ii) Implement gradient descent with momentum for the above two datasets. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent
     (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an epsilon-neighborhood of the minimizer for both datasets. Choose            epsilon = 0.001. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value         oftheta,momentum,etc.

# Taken:

      number of epochs = 1000
      epsilon = 0.001
      learning_rate = 0.001
      momentum = 0.9

# Observation:

            #Dataset 1:
        
            Full Batch Gradient Descent with Momentum:
            Parameters (theta): [ 0.52210738 99.99454276]
            Final Loss: 0.8182701377879564
            
            Stochastic Gradient Descent with Momentum:
            Parameters (theta): [ 0.95621419 99.98579381]
            Final Loss: 0.5961058206021077
            
            
            #Dataset 2:
            
            Full Batch Gradient Descent with Momentum:
            Parameters (theta): [ 3.63140271 57.0637155 ]
            Final Loss: 2.7355191432156
            
            Stochastic Gradient Descent with Momentum:
            Parameters (theta): [ 3.94917427 59.50983949]
            Final Loss: 0.6051241046603109
    
(A) Compare the average number of steps taken with gradient descent (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an             epsilon-neighborhood of the minimizer for both datasets.

# Dataset 1:

    Full batch gradient descent(with momentum) steps: 165
    Stochastic gradient descent(with momentum) steps: 460

    #Stochastic gradient descent takes more epochs to converge.

    #Average number of steps taken to converge:
    Gradient Descent with Momentum (Full Batch): 165.0
    Gradient Descent with Momentum (Stochastic): 326.5
    Full-Batch Gradient Descent(without Momentum): 1000.0
    Stochastic Gradient Descent(without Momentum): 63.01

# Comparison with and without Momentum:

* Both full-batch and stochastic gradient descent with momentum outperform their non-momentum counterparts in terms of average steps to converge. This indicates that momentum helps accelerate convergence for both optimization methods.

* Full-batch gradient descent with momentum shows a significant improvement in convergence compared to without momentum, indicating the effectiveness of momentum in improving convergence speed for full-batch optimization.

* Stochastic gradient descent with momentum also benefits from the addition of momentum, but it still requires more steps to converge compared to full-batch gradient descent with momentum. This could be due to the stochastic nature of the optimization process, which introduces more variability in the convergence behavior.

# Comparison between Full Batch and Stochastic Gradient Descent with Momentum:

* Full-batch gradient descent with momentum takes fewer steps to converge compared to stochastic gradient descent with momentum for Dataset 1. This suggests that for Dataset 1, the full-batch approach is more efficient in utilizing momentum to achieve convergence.

* The lower steps to converge for full-batch gradient descent with momentum also result in a lower average compared to stochastic gradient descent with momentum.

    
# Dataset 2:

    Full batch gradient descent(with momentum) steps: 1000
    Stochastic gradient descent(with momentum) steps: 49

    #Full-Batch gradient descent takes more epochs to converge.

    #Average number of steps taken to converge:
    Gradient Descent with Momentum (Full Batch): 1000.0
    Gradient Descent with Momentum (Stochastic): 48.98
    Full-Batch Gradient Descent(without Momentum): 1000.0
    Stochastic Gradient Descent(without Momentum): 417.88

# Comparison with and without Momentum:

* Similar to Dataset 1, both full-batch and stochastic gradient descent with momentum outperform their non-momentum counterparts in terms of average steps to converge. This indicates that momentum is effective in improving convergence speed for both optimization methods.

* Full-batch gradient descent with momentum shows no improvement in convergence compared to without momentum, suggesting that momentum does not provide significant benefits for full-batch optimization in this dataset.

# Comparison between Full Batch and Stochastic Gradient Descent with Momentum:

* Stochastic gradient descent with momentum takes fewer steps to converge compared to full-batch gradient descent with momentum for Dataset 2. This indicates that for Dataset 2, the stochastic approach is more efficient in utilizing momentum to achieve convergence.

* The lower steps to converge for stochastic gradient descent with momentum also result in a lower average compared to full-batch gradient descent with momentum.


# Overall:

The results show that the effectiveness of momentum varies depending on the dataset and the optimization method used (full-batch or stochastic).

* In Dataset 1, full-batch gradient descent with momentum outperforms stochastic gradient descent with momentum in terms of convergence speed, while in Dataset 2, stochastic gradient descent with momentum shows better convergence.

* Momentum generally improves convergence speed for both full-batch and stochastic gradient descent, but its effectiveness may vary depending on the optimization problem's characteristics.

















