import streamlit as st
import numpy as np
import pandas as pd
from ATC_Predictor_BWfun import Attention
from keras.models import *
from keras.layers import *
import keras.backend.tensorflow_backend as K
import keras
import pickle
from keras.preprocessing import sequence
import tensorflow as tf

st.set_page_config(
    page_title="Medication Autoencoder",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.sidebar.title("Introduction")
st.sidebar.markdown("A deep learning model that predicts the ATC code of medication.")


@st.cache(
    hash_funcs={keras.engine.training.Model: id},
    allow_output_mutation=True,
)
def load_model():
    """
    Load model to cache
    """
    with open("char_attn_lstm_model.json") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(
        loaded_model_json, custom_objects={"Attention": Attention()}
    )
    model.load_weights("char_attn_lstm_model.h5")
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model._make_predict_function()
    graph = tf.get_default_graph()
    return model, graph


model, graph = load_model()
#  write a try catch here to prevent excel files
def run_model(data) -> pd.DataFrame:
    """Run model

    Args:
        data ([DataFrame]): input file

    Returns:
        [DataFrame]: predictions
    """
    # keras.backend.get_session().run(tf.local_variables_initializer())
    # keras.backend.get_session().run(tf.global_variables_initializer())
    # this is a keras/tf bug for this version
    keras.backend.get_session().run(tf.initialize_all_variables())
    global model
    TEXT_COLUMN = data.columns[0]
    data = data.dropna(subset=[TEXT_COLUMN])  # Ensure no missing data in the input data
    # Load tokenizer
    with open("new_tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    mapped_list = tokenizer.texts_to_sequences(data[TEXT_COLUMN].tolist())
    max_review_length = 256
    mapped_list = sequence.pad_sequences(mapped_list, maxlen=max_review_length)
    y_score_prob = model.predict(mapped_list)
    y_score = np.argmax(y_score_prob, axis=1)
    data["PREDICTED_ATC_INDEX"] = y_score
    lookup = pd.read_csv("atc_dictionary.csv", header=0, encoding="latin-1")
    data2 = pd.merge(
        data, lookup, how="left", left_on=["PREDICTED_ATC_INDEX"], right_on=["INDEX"]
    )
    data2 = data2[[TEXT_COLUMN, "ATC Code"]]
    return data2


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        label="Go to", options=["Home", "User Guide", "Try it out!"]
    )

    if selection == "Home":
        st.title("Medication Autoencoder using Attention-based LSTM")
    if selection == "User Guide":
        st.title("User Guide")
        st.markdown(">1. Prepare an input file that looks like:")
        st.write(
            pd.read_csv(
                "Sample_Input.csv",
                header=0,
                na_values=["", "-", "."],
                encoding="latin-1",
            ).head(5)
        )
        st.markdown(
            ">:warning: Input file should only consists of **one** column, with drug and dosage information inside."
        )
        st.markdown(">2. Upload/Drag input file.")
        st.markdown(
            ">3. Wait for prediction to be generated. Typically takes more than 10s because the model is huge."
        )
    if selection == "Try it out!":
        uploaded_file = st.file_uploader("Please upload an input file", type="csv")
        # run_model_button = st.button("Run Model!")
        # there's a streamlit bug with file uploader and interactive widgets, so
        # removing the upload button first
        if uploaded_file is not None:
            input_file = pd.read_csv(
                uploaded_file, header=0, na_values=["", "-", "."], encoding="latin-1"
            )

            if len(input_file.columns) == 1:
                global graph
                with st.spinner("Generating predictions..."):
                    with graph.as_default():
                        predictions = run_model(input_file)
                        st.markdown(">First 5 rows of Predictions:")
                        st.write(predictions.head(5))
            else:
                st.error("Input file must only have one column!")

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by IHiS. You can learn more about us at
        [ihis.com.sg](https://ihis.com.sg).
"""
    )


main()