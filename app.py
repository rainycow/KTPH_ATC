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
import base64
from datetime import datetime
import math

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


def download_link(object_to_download, download_filename):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}"><input type="button" value="Download file"></a>'


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
                        st.markdown(">First 10 rows of Predictions:")
                        PAGE_SIZE = 10
                        page_number = st.number_input(
                            label="Page Number",
                            min_value=1,
                            max_value=math.ceil(len(predictions) / PAGE_SIZE),
                            step=1,
                        )
                        current_start = (page_number - 1) * PAGE_SIZE
                        current_end = page_number * PAGE_SIZE
                        st.write(predictions[current_start:current_end])
                        TIME = datetime.now().strftime("%Y%m%d_%I%M")
                        FILENAME = TIME + "predictions.csv"
                        st.markdown(
                            download_link(predictions, FILENAME), unsafe_allow_html=True
                        )

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