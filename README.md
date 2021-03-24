# kernel-dna-classification
Predicting whether a DNA sequence region is a binding site to a specific transcription factor (Data Challenge).


# Setup
## 1. Data
For the code to run properly the data must be put in a folder called `data` and split into `train` and `test` folders, following this structure

```
data  
│
└───train
│   │   Xtr0_mat100.csv
│   │   Xtr0.csv
│   │   Ytr0.csv
│   │   Xtr1_mat100.csv
│   │   Xtr1.csv
│   │   Ytr1.csv
│   │   Xtr2_mat100.csv
│   │   Xtr2.csv
│   │   Ytr2.csv
│   └─
│   
└───test
    │   Xte0_mat100.csv
    │   Xte0.csv
    │   Xte1_mat100.csv
    │   Xte1.csv
    │   Xte2_mat100.csv
    │   Xte2.csv
    └─
```

## 2. Reproduce the submission file
To reporduce the submission file you can run the script in `start.py` 
```bash
python start.py
```
this command will create the submission file in the folder `data`.


# Code
## 1. Kernels
The kernels along with their helper functions are defined in the file `kernels.py`. For now, you can find the linear kernel, the gaussian kernel and the spectrum kernel.

## 2. Models
The models are defined as classes in the file `models.py`. You can find Kernel Ridge Regression, Kernel Logistic Regression and Kernel Support Vector Machines.

There is an extra file called `test_models.py` to test the aforementioned models on simple cases.

## 3. Tuning the Hyperparameters
The code used to tune the hyperparameters is in `utils.py`.