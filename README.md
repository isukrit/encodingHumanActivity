# Deep ConvLSTM with self-attention for human activity decoding using wearables.
In this git repository we implement the proposed novel architecture for encoding human activity data for body sensors. The proposed model encodes the sensor data in both the spatial domain (whereby it selects important sensors) and the time domain (whereby it selects important time points). 

![proposed Architecture](https://github.com/isukrit/encodingHumanActivity/blob/master/sensors_architecture.PNG)

We have added codes for two of models that we tested:
- Baseline CNN + LSTM model
- Proposed model with self-attention

## Setup

`pip3 install -r requirements.txt`

## Guide to run the code

1. The data for the subjects resides in the `data` directory. You can create subdir for the arrangement and the dataset.npy file as `data/arrangement/dataset.npz`. We used the processed data from [here](https://github.com/arturjordao/WearableSensorData/tree/master/data) and put it in the data folder. For example: 'data/LOTO/MHEALTH.npz' file depicts the leave one trial out (LOTO) for the MHEALTH dataset. 

2. The codes reside in the  `codes` folder. 
* `codes/model_baseline`: This folder contains the Python code for training a baseline model containing CNN and RNN layers. You can run the code by running the following command:
```
python3 model_baseline.py
```

* `codes/model_proposed`: This folder contains the Python code for training the proposed model built using Keras and Tensorflow. Self-attention was coded from [here](https://github.com/uzaymacar/attention-mechanisms). You can run the code by running the following command:

```
python3 model_with_self_attn.py
```

