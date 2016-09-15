# Chord Recognition

## classic methods

- pitch *chroma* vectors extracted from the audio as input *feature* for chord detection

- Hidden Markov Models (HMMs) to stabilize chord classification

  **train HMM with chroma features**

## some recent methods for training models 

- DNN

- RBM

- DBN

- CNN

- bottleneck architecture (reduce overfitting)

  **pre-train (RBM) + fine-tune (BP)**

  **shared weights** for CNN

## before training

- CQT (Constant Q Transform) + PCA (Principal Component Analysis)
- DFT
- splice and filter

## after training

- SVM

- HMM (standard)

  **classification**

## evaluation method

WCSR (Weighted Chord Symbol Recall): the total duration of segments with correct prediction

