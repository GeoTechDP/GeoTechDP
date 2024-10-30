# Differential Privacy in Geotechnical Engineering

# Usage
**(1) Install**

Run the following commands:
```
# Install TensorFlow and Keras
$ pip install tensorflow==2.15.0
$ pip install keras==2.15.0

# Install TensorFlow Privacy
$ pip install tensorflow-privacy==0.9.0

# Install scikit-learn
$ pip install scikit-learn==1.4.2
```

**(2) Generate Datasets**

Run the following commands:
```
# Generate 150000 CPT samples
$ python Gen_1DRF_allfunc_main.py
$ mv output_with_sof.csv output_with_sof_A3_T50.csv
```

**(3) Evaluate SGD/DPSGD/Adam**

Run the following commands:
```
# Evaluate Adam(0)/DPSGD(1)/SGD(2) using the CPT samples (#epoch=100, C=0.1, sigma=1.5)
$ python CNN2021_Regression_DPSGD.py 0 100 0.1 1.5
$ python CNN2021_Regression_DPSGD.py 1 100 0.1 1.5
$ python CNN2021_Regression_DPSGD.py 2 100 0.1 1.5

# Evaluate Adam(0)/DPSGD(1)/SGD(2) using data_8_3972.csv (#epoch=100, C=0.1, sigma=1.5)
$ python FCNN_DPSGD.py 0 100 0.1 1.5
$ python FCNN_DPSGD.py 1 100 0.1 1.5
$ python FCNN_DPSGD.py 2 100 0.1 1.5
```

# References
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)
- [Implement Differential Privacy with TensorFlow Privacy](https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy)
