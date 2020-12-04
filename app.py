import streamlit as st
import numpy as np
import pandas as pd
from ATC_Predictor_BWfun import Attention
from keras.models import *
from keras.layers import *
import keras.backend.tensorflow_backend as K
import keras

st.set_page_config(
    page_title="Medication Autoencoder",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.sidebar.title("Introduction")
st.sidebar.markdown("A deep learning model that predicts the ATC code of medication.")
# st.sidebar.title("To Use")


@st.cache(
    hash_funcs={keras.engine.training.Model: id},
    persist=True,
    allow_output_mutation=True,
)
def load_model():
    from keras.models import load_model

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
    session = K.get_session()
    return model, session


model = load_model()
#  write a try catch here to prevent excel files


# confidence_value = st.sidebar.slider("Confidence:", 0.0, 1.0, 0.5, 0.1)
# if uploaded_file:
#     st.sidebar.info("Uploaded image:")
#     grad_cam_button = st.sidebar.button("Grad CAM")
#     patch_size_value = st.sidebar.slider("Patch size:", 10, 90, 20, 10)
#     occlusion_sensitivity_button = st.sidebar.button("Occlusion Sensitivity")
#     image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     orig = image.copy()
#     (h, w) = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

# st.sidebar.title("About")
# st.sidebar.image("assets/IHiSlogo.jpg", width=100)


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        label="Go to", options=["Home", "User Guide", "Try it out!"]
    )

    if selection == "Home":
        st.title("Medication Autoencoder using Attention-based LSTM")
    if selection == "User Guide":
        st.title("To be filled in")
    if selection == "Try it out!":
        uploaded_file = st.file_uploader("Please upload an input file", type="csv")

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by IHiS. You can learn more about us at
        [datamodelsanalytics.com](https://datamodelsanalytics.com).
"""
    )


main()