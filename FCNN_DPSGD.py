import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement

#from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers.legacy import Adam, SGD
#import joblib
#from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import random
import os
import sys

# Set random seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


################################### Parameters ###################################

if len(sys.argv) == 1:
    #sys.argv = ["CNN2021_Regression_DPSGD.py", "D", "ep", "c", "n"]
    #sys.argv = ["CNN2021_Regression_DPSGD.py", "1", "10", "1", "1.5"]
    #sys.argv = ["CNN2021_Regression_DPSGD.py", "0", "100", "1", "1"]
    #sys.argv = ["CNN2021_Regression_DPSGD.py", "2", "100", "1", "1"]
    # sys.argv = ['FCNN_DPSGD.py', '1', '400', '1', '1']
    sys.argv = ['FCNN_DPSGD.py', '1', '100', '0.1', '3']
    print("Usage:",sys.argv[0],"[D (2:SGD, 1:DP-SGD, 0:Adam)] [e (#epochs)] [c (clip size)] [n (noise multiplier)]")
    print('now using default parameters')
elif len(sys.argv) < 5:
    print("Usage:",sys.argv[0],"[D (2:SGD, 1:DP-SGD, 0:Adam)] [e (#epochs)] [c (clip size)] [n (noise multiplier)]")
    sys.exit(0)

# Dataset path
#PATH = "output_with_sof_A3_T5.csv"
#PATH = "output_with_sof_A3_T10.csv"
#PATH = "output_with_sof_A3_T20.csv"
#PATH = "output_with_sof_A3_T30.csv"
#PATH = "output_with_sof_A3_T40.csv"
# PATH = "output_with_sof_A3_T50.csv"
PATH = "./data_8_3972.csv"

# Proportion of training set and test set
TEST_SIZE = 0.2
# Batch size
BATCH_SIZE = 32
# Learning rate
LEARN_RATE = 0.001

# DP-SGD (2:SGD, 1:DP-SGD, 0: Adam)
DPSGD = int(sys.argv[1])
#DPSGD = 0

# Number of iterations
EPOCHS = int(sys.argv[2])
#EPOCHS = 100

# DP-SGD parameters
L2_NORM_CLIP = float(sys.argv[3])    # The l2 clip norm
#L2_NORM_CLIP = 1.5    # The l2 clip norm

NOISE_MULTIPLIER = float(sys.argv[4])  # The ratio of the Gaussian noise stddev to the l2 clip norm at each round
#NOISE_MULTIPLIER = 1.3  # The ratio of the Gaussian noise stddev to the l2 clip norm at each round

NUM_MICROBATCHES = 1   # "NUM_MICROBATCHES = 1" reduces to the standard DP-SGD w/o microbatching

# Number of points to plot
PLOT_NUM = 1000

# Result file
if len(sys.argv) >= 6:
    if sys.argv[5] == "clip":
        RESFILE_EPS = "out/res_eps_mse_D" + sys.argv[1] + "_c" + sys.argv[3] + ".csv"
    elif sys.argv[5] == "noise":
        RESFILE_EPS = "out/res_eps_mse_D" + sys.argv[1] + "_n" + sys.argv[4] + ".csv"
else:
    RESFILE_EPS = "out/res_eps_mse_D" + sys.argv[1] + "_c" + sys.argv[3] + ".csv"
RESFILE_MSE = "out/res_epochs_mse_D" + sys.argv[1] + "_e" + sys.argv[2] + "_c" + sys.argv[3] + "_n" + sys.argv[4] + ".pdf"
RESFILE_QU_TR = "out/res_QU_train_D" + sys.argv[1] + "_e" + sys.argv[2] + "_c" + sys.argv[3] + "_n" + sys.argv[4] + ".pdf"
RESFILE_QU_TE = "out/res_QU_test_D" + sys.argv[1] + "_e" + sys.argv[2] + "_c" + sys.argv[3] + "_n" + sys.argv[4] + ".pdf"

# Common plot size and font size parameters
FIG_SIZE = (8, 6)  # Uniform size for all PDFs (width, height)
FONT_SIZE = 20     # Default font size; adjust as needed or pass via a command line argument
DO_PLOT = False
# DO_PLOT = True

##################################################################################


def extract_epsilon(eps_str):
    # Split the string into lines
    lines = eps_str.splitlines()
    
    # Iterate through the lines to find the "User-level DP with" line
    for i, line in enumerate(lines):
        if "User-level DP with" in line:
            # The target line is 3 lines after this one
            target_line = lines[i + 3]
            # Extract the number at the end of the target line
            epsilon_str = target_line.split()[-1]
            return float(epsilon_str)
    
    # Return None if the pattern was not found
    return None


if BATCH_SIZE % NUM_MICROBATCHES != 0:
  raise ValueError('Batch size should be an integer multiple of the number of microbatches')


############################### 1. Load the dataset ###############################
# Load the dataset
# data = pd.read_csv(PATH, header=None).T.values
# np.random.shuffle(data)

# target = data[:,100]
# train = data[:,:100]

# scaler = StandardScaler()
# train = scaler.fit_transform(train)

# # Split dataset
# x_train,x_test,y_train,y_test = train_test_split(train,target,test_size=TEST_SIZE, random_state=1)

# x_train = x_train.reshape(len(x_train),100,1)
# x_test = x_test.reshape(len(x_test),100,1)

# Load the dataset
data = pd.read_csv(PATH, header=0)

# encoding and standardization
data = pd.get_dummies(data, columns=['label'], prefix='', prefix_sep='') # labelはdropされる
data[['Main_low', 'Main_mid', 'Main_up']] = data[['Main_low', 'Main_mid', 'Main_up']].astype(int)
data[data.select_dtypes(include=[np.float64]).columns] = (
    data.select_dtypes(include=[np.float64]) - data.select_dtypes(include=[np.float64]).mean()
) / data.select_dtypes(include=[np.float64]).std()

# Split dataset by borhole location
unique_xy = data[['X', 'Y']].drop_duplicates()
train_xy = unique_xy.sample(frac=0.8, random_state=seed_value)
test_xy = unique_xy.drop(train_xy.index)

def pick_match_rows(df, xy):
    return df[df[['X', 'Y']].apply(tuple, axis=1).isin(xy.apply(tuple, axis=1))]
train_data = pick_match_rows(data, train_xy)
test_data = pick_match_rows(data, test_xy)

def split_xy(df):
    return df.drop('qu', axis=1).values, df['qu'].values
x_train, y_train = split_xy(train_data)
x_test, y_test = split_xy(test_data)

# reshape
# x_train = np.expand_dims(x_train, axis=2)
# x_test = np.expand_dims(x_test, axis=2)

####################### 2. Start stacking each neural layer #######################
# Start stacking each neural layer (cf. Table 1)
# model = Sequential()

# model.add(Conv1D(filters=64,kernel_size=5, input_shape=(100,1), activation="relu") ) #kernel_regularizer=regularizers.l2(0.01)

# model.add(MaxPooling1D(pool_size=2))

# model.add(Conv1D(filters=64,kernel_size=5, activation="relu") )

# model.add(MaxPooling1D(pool_size=2))

# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('sigmoid'))
# model.add(Dense(32))

# model.add(Activation('sigmoid'))

# model.add(Dense(1))

def make_fcnn(in_dim, out_dim, mid_dims=[32, 32, 32]):
    model = Sequential() #init
    
    model.add(Dense(mid_dims[0], input_dim=in_dim, activation='relu')) # input
    for units in mid_dims[1:]: # hidden
        model.add(Dense(units, activation='relu'))
    model.add(Dense(out_dim)) # output
    
    return model

model = make_fcnn(in_dim=x_train.shape[1], out_dim=1,
                  mid_dims=[32, 32, 32])


############################# 3. Define an optimizer ##############################
# Define an optimizer (2:SGD, 1:DP-SGD, 0: Adam)
if DPSGD==1:
    # Define the DP-SGD optimizer
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=L2_NORM_CLIP,
        noise_multiplier=NOISE_MULTIPLIER,
        num_microbatches=NUM_MICROBATCHES,
        learning_rate=LEARN_RATE
    )
    # Error function: mse, optimizer: DP-SGD
    model.compile(loss='mse', optimizer=optimizer)

    # Compute DP-SGD privacy epsilon
    eps_str = compute_dp_sgd_privacy_statement(
        number_of_examples=len(x_train),
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        noise_multiplier=NOISE_MULTIPLIER,
        delta=1e-4,
#        used_microbatching=True,
        used_microbatching=False,   #   w/o microbatching
        max_examples_per_user=1
    )
    eps = extract_epsilon(eps_str)
#    print(eps)
elif DPSGD==2:
    optimizer = SGD(learning_rate=LEARN_RATE)
    # Error function: mse, optimizer: adam
    model.compile(loss='mse',
                  optimizer=optimizer,
                )
    eps = 0;
else:
    optimizer = Adam(learning_rate=LEARN_RATE)
    # Error function: mse, optimizer: adam
    model.compile(loss='mse',
                  optimizer=optimizer,
                )
    eps = 0; 


######################## 4. Model training and prediction #########################
# Model training
loss_and_metrics = model.fit(x_train, y_train,
        batch_size=BATCH_SIZE,epochs=EPOCHS,
        validation_data = (x_test,y_test),
        verbose=0)
print("-------------------------")
print("Model training completed!")
print("-------------------------")

# Plot loss (cf. Figure 4, left)
if DO_PLOT:
    plt.figure(figsize=FIG_SIZE)
    plt.plot(loss_and_metrics.history['loss'],label='Training Data')
    plt.plot(loss_and_metrics.history['val_loss'],label='Testing Data')
    plt.xlabel("Epochs", fontsize=FONT_SIZE)
    plt.ylabel("MSE", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=FONT_SIZE)  # Set font size for y-axis tick labels
    plt.legend(fontsize=FONT_SIZE)
    plt.savefig(RESFILE_MSE)
    # plt.show()
train_loss = loss_and_metrics.history['loss']
test_loss = loss_and_metrics.history['val_loss']

#print("Model saved successfully!")
#model.save("model123.h5")

# Prediction
x_trainprediction = model.predict(x_train)
x_testprediction = model.predict(x_test)


################################ 5. Calculate MSE #################################
# If they are tensors, convert them to numpy arrays first
y_train_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
x_trainprediction_np = x_trainprediction.numpy() if hasattr(x_trainprediction, 'numpy') else x_trainprediction

y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else y_test
x_testprediction_np = x_testprediction.numpy() if hasattr(x_testprediction, 'numpy') else x_testprediction

# Ensure y_test_np, x_testprediction_np, y_train_np, and x_trainprediction_np are 1D arrays
y_train_np = y_train_np.flatten()
x_trainprediction_np = x_trainprediction_np.flatten()
y_test_np = y_test_np.flatten()
x_testprediction_np = x_testprediction_np.flatten()

# Calculate MSE
mse_train = mean_squared_error(y_train_np, x_trainprediction_np)
mse_test = mean_squared_error(y_test_np, x_testprediction_np)

# Output the parameters, epsilon, and MSE (cf. Figures 5 and 6)
# Check if the file exists and if it's empty
if not os.path.exists(RESFILE_EPS) or os.stat(RESFILE_EPS).st_size == 0:
    with open(RESFILE_EPS, "w") as f:
        f.write("DPSGD,epochs,l2_norm_clip,noise_multiplier,epsilon,mse_train,mse_test\n")

f = open(RESFILE_EPS, "a")
f.write(f"{DPSGD},{EPOCHS},{L2_NORM_CLIP},{NOISE_MULTIPLIER},{eps},{mse_train},{mse_test}\n")
f.close()


######################## 6. Plot actual and estimated QUs ########################
# Calculate correlation coefficient R for train and test sets
# correlation_r_train = np.corrcoef(y_train_np, x_trainprediction_np)[0, 1]
correlation_r_test = np.corrcoef(y_test_np, x_testprediction_np)[0, 1]

# # Randomly select PLOT_NUM points to plot from training set
# if len(y_train_np) > PLOT_NUM:
#     indices_train = np.random.choice(len(y_train_np), PLOT_NUM, replace=False)
#     y_train_plot = y_train_np[indices_train]
#     x_trainprediction_plot = x_trainprediction_np[indices_train]
# else:
#     y_train_plot = y_train_np
#     x_trainprediction_plot = x_trainprediction_np

# # Create the plot for the training set
# plt.figure(figsize=FIG_SIZE)
# #plt.scatter(y_train_np, x_trainprediction_np, color='green', marker='o')
# plt.scatter(y_train_plot, x_trainprediction_plot, color='green', marker='o')
# #plt.title('Actual vs Predicted (Training Set)', fontsize=FONT_SIZE)
# plt.xlabel('Actual QU', fontsize=FONT_SIZE)
# plt.ylabel('Estimated QU', fontsize=FONT_SIZE)
# plt.xticks(fontsize=FONT_SIZE)  # Set font size for x-axis tick labels
# plt.yticks(fontsize=FONT_SIZE)  # Set font size for y-axis tick labels
# plt.grid(True)

# # Display MSE and correlation coefficient R for the training set on the plot, adjusting the position
# textstr_train = f'MSE: {mse_train:.4f}\nR: {correlation_r_train:.4f}'
# plt.gcf().text(0.15, 0.75, textstr_train, fontsize=FONT_SIZE, bbox=dict(facecolor='white', alpha=0.5))

# # Save the plot as a PDF file
# plt.savefig(RESFILE_QU_TR)
# plt.show()

# Randomly select PLOT_NUM points to plot from test set
if len(y_test_np) > PLOT_NUM:
    indices_test = np.random.choice(len(y_test_np), PLOT_NUM, replace=False)
    y_test_plot = y_test_np[indices_test]
    x_testprediction_plot = x_testprediction_np[indices_test]
else:
    y_test_plot = y_test_np
    x_testprediction_plot = x_testprediction_np

if DO_PLOT:
    # Create the plot for the test set (cf. Figure 4, right)
    plt.figure(figsize=FIG_SIZE)
    #plt.scatter(y_test_np, x_testprediction_np, color='blue', marker='o')
    plt.scatter(y_test_plot, x_testprediction_plot, color='blue', marker='o')
    plt.axis('equal')
    #plt.title('Actual vs Predicted (Testing Set)', fontsize=FONT_SIZE)
    plt.xlabel('Actual QU', fontsize=FONT_SIZE)
    plt.ylabel('Estimated QU', fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=FONT_SIZE)  # Set font size for y-axis tick labels
    plt.grid(True)

    # Display MSE and correlation coefficient R for the test set on the plot, adjusting the position
    textstr_test = f'MSE: {mse_test:.4f}\nR: {correlation_r_test:.4f}'
    plt.gcf().text(0.15, 0.75, textstr_test, fontsize=FONT_SIZE, bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot as a PDF file
    plt.savefig(RESFILE_QU_TE)
    # plt.show()