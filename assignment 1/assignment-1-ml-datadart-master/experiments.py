import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Function to create fake data
def create_fake_data(N, M):
    # Create a DataFrame with N samples and M binary features
    data = pd.DataFrame(np.random.randint(2, size=(N, M)), columns=[f'feature_{i}' for i in range(M)])
    data['target'] = np.random.randint(2, size=N)  # Binary target variable
    return data

# Function to calculate average time taken by fit() and predict() for different N and M
def run_experiments():
    N_values = [100, 500, 1000, 2000] 
    M_values = [10, 50, 100, 200]  

    for criterion in ['gini', 'entropy']:
        fit_times = []
        predict_times = []

        for N in N_values:
            for M in M_values:
                total_fit_time = 0
                total_predict_time = 0

                for _ in range(num_average_time):
                    # Create fake data
                    data = create_fake_data(N, M)

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

                    # Initialize Decision Tree
                    dt = DecisionTreeClassifier(criterion=criterion)

                    # Measure fit time
                    start_time = time.time()
                    dt.fit(X_train, y_train)
                    total_fit_time += time.time() - start_time

                    # Measure predict time
                    start_time = time.time()
                    dt.predict(X_test)
                    total_predict_time += time.time() - start_time

                # Calculate average fit and predict times
                avg_fit_time = total_fit_time / num_average_time
                avg_predict_time = total_predict_time / num_average_time

                fit_times.append(avg_fit_time)
                predict_times.append(avg_predict_time)

        # Plot the results
        plot_results(N_values, M_values, fit_times, predict_times, criterion)

# Function to plot the results
def plot_results(N_values, M_values, fit_times, predict_times, criterion):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, M in enumerate(M_values):
        plt.plot(N_values, fit_times[i * len(N_values):(i + 1) * len(N_values)], label=f'M={M}')
    plt.title(f'{criterion.capitalize()} Decision Tree - Learning Time')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Average Time (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i, M in enumerate(M_values):
        plt.plot(N_values, predict_times[i * len(N_values):(i + 1) * len(N_values)], label=f'M={M}')
    plt.title(f'{criterion.capitalize()} Decision Tree - Prediction Time')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Average Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run the experiments
run_experiments()
