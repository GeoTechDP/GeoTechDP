import numpy as np
import pandas as pd

# Fix the random seed for reproducibility
np.random.seed(42)

# Define constants
dz = 0.1
D = 10
sig = 1
n = round(D / dz)
N = 100
SOF1 = 0.1
SOF2 = 3

# Specify which functions to use
# (1: Single Exponential, 2: Second-order Markov, 3: Squared Exponential, 4: Binary Noise, 5: Cosine Exponential)
#Func = [1, 2, 3, 4, 5]
#Func = [1]
#Func = [2]
Func = [3]
#Func = [4]
#Func = [5]

# Number of trials to perform
#Trial = 1
#Trial = 5
#Trial = 10
#Trial = 20
#Trial = 30
#Trial = 40
Trial = 50

# List to store results
All = []

SOF_cand = np.arange(0.1, 3.1, 0.1)
SOF_cand = np.round(SOF_cand, 1)

# Loop for SOF
for s in SOF_cand:
    for func in Func:
        for _ in range(Trial):  # Repeat based on the number of trials
            cov_max = np.zeros((n, n))  # Initialize covariance matrix
            K = 10**(-14) * np.eye(n)  # Additional matrix for numerical stability of the SqExp model

            # Calculate the covariance matrix based on the selected function
            for i in range(n):
                for j in range(n):
                    if func == 1:
                        # Single Exponential
                        cov_max[i, j] = np.exp(-2 * (abs(i - j) / (1 / dz)) / s)

                    elif func == 2:
                        # Second-order Markov
                        cov_max[i, j] = (1 + 4 * (abs(i - j) / (1 / dz)) / s) * np.exp(-4 * (abs(i - j) / (1 / dz)) / s)

                    elif func == 3:
                        # Squared Exponential
                        cov_max[i, j] = np.exp(-np.pi * (abs(i - j) / (1 / dz)) ** 2 / (s ** 2)) + K[i, j]

                    elif func == 4:
                        # Binary Noise
                        if abs(i - j) / (1 / dz) > s:
                            cov_max[i, j] = 0  # Binary noise BN model, condition when abs(i-j)/(1/dz) > s
                        else:
                            cov_max[i, j] = 1 - (abs(i - j) / (1 / dz)) / s  # Binary noise BN model, otherwise

                    elif func == 5:
                        # Cosine Exponential
                        cov_max[i, j] = np.cos((abs(i - j) / (1 / dz)) / s) * np.exp(-(abs(i - j) / (1 / dz)) / s)

            # Perform Cholesky decomposition
            L = np.linalg.cholesky(cov_max)

            # Generate random normal numbers and multiply with matrix L
            w = sig * L @ np.random.randn(n, N)

            # Save the result
            All.append(w)

# Convert list of arrays to a single 2D array
All_array = np.hstack(All)  # Horizontally stack the list of arrays into one large 2D array

# Convert to a pandas DataFrame
df = pd.DataFrame(All_array)

# Create an array that repeats SOF_cand for each trial and function
SOF_row = np.repeat(SOF_cand, len(Func) * Trial * N)

# Ensure SOF_row has the same number of columns as df
if len(SOF_row) != df.shape[1]:
    raise ValueError(f"Length of SOF_row ({len(SOF_row)}) does not match number of columns in df ({df.shape[1]})")

# Add SOF values as the last row
SOF_row_df = pd.DataFrame([SOF_row], columns=df.columns)

# Append the SOF_row to the DataFrame
df = pd.concat([df, SOF_row_df], ignore_index=True)

# Save the DataFrame to a CSV file without the index
df.to_csv('output_with_sof.csv', index=False, header=False)

# Display result
print('Finished and saved to output_with_sof.csv')
