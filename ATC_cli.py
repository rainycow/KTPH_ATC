import sys
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed

set_random_seed(1234)
from keras.models import *
from keras.layers.embeddings import Embedding
from keras.layers import *
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle
import random as rn

rn.seed(1)
from numpy.random import seed

seed(1234)
from keras.callbacks import EarlyStopping

# below is to allow GPU to be use in other jupyter notebook
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# LOAD ATTENTION LAYER
class Attention(Layer):
    def __init__(
        self,
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        bias=True,
        **kwargs
    ):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = 256
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            (input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(
            K.dot(
                K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))
            ),
            (-1, step_dim),
        )

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


FILENAME = sys.argv[1]
data = pd.read_csv(
    "Sample_Input.csv", header=0, na_values=["", "-", "."], encoding="latin-1"
)  # IMPORT UR DATA HERE

# column name
TEXT_COLUMN = data.columns[0]
data = data.dropna(
    subset=[TEXT_COLUMN]
)  # IF YOU HAVE NA VALUE, my dataframe contains a few cols, ORIG is the col with description
data = pd.DataFrame(data[TEXT_COLUMN])
# DROP DUPLIDATES TO AVOID MEMORY ISSUE
data = data.drop_duplicates()

# LOAD DICTIONARY
with open("new_tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

mapped_list = tokenizer.texts_to_sequences(data[TEXT_COLUMN].tolist())
max_review_length = 256
mapped_list = sequence.pad_sequences(mapped_list, maxlen=max_review_length)

# LOAD MODEL
from keras.models import load_model

json_file = open("char_attn_lstm_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
tmp_model = model_from_json(
    loaded_model_json, custom_objects={"Attention": Attention()}
)
tmp_model.load_weights("char_attn_lstm_model.h5")
tmp_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

y_score_prob = tmp_model.predict(mapped_list)
y_score = np.argmax(y_score_prob, axis=1)
data["PREDICTED_ATC_INDEX"] = y_score

lookup = pd.read_csv("atc_dictionary.csv", header=0, encoding="latin-1")
data2 = pd.merge(
    data, lookup, how="left", left_on=["PREDICTED_ATC_INDEX"], right_on=["INDEX"]
)

data2 = data2[[TEXT_COLUMN, "ATC Code"]]
data2.to_csv("predicted_atc_ty.csv", index=False)
