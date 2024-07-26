import os
import sys
sys.path.append(os.path.join(os.getcwd(),'app'))

import streamlit as st

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import set_background, load_selected_features

import warnings
warnings.filterwarnings("ignore")


st.markdown("""
<style>
.stDeployButton
{
    visibility : hidden;
}
.st-emotion-cache-czk5ss.e16jpq800
{
    visibility : hidden;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title='agri-food emissions',
    page_icon='📈'
    )

# set_background('background_img.jpg')

st.title("Agri Food Emissions Prediction")

features = load_selected_features(os.path.join('notebook','selected_features.json'))

form = st.form("Form 1")

col1,col2,col3,col4 = form.columns(4)

with col1:
    a = st.text_input(features[0])

btn = form.form_submit_button("Submit")

if btn : 
    pass




# # classification
# if st.button('Results'):
#     input_ref = ["01","02","03","04","05","06","07","08","09","10","11","12","13"]
#     input_list = [crim, zn, indus, chas, nox, rm, age, dis, rad, chas, ptratio, b, lstat]
#     missing_values = [input_ref[i] for i, val in enumerate(input_list) if val is None or val == '']

#     if missing_values:
#         st.write("## Missing Values")
#         for missing_value in missing_values:
#             st.write(f"### Column '{missing_value}' missing.")
#         st.write("## Please refresh and enter the values again")
    
#     else: 
#         result = predict([float(crim),float(zn),float(indus),float(chas),float(nox),float(rm),float(age),float(dis),float(rad),float(chas),float(ptratio),float(b),float(lstat)],model)

#         # show results
#         st.write("## Result")
#         st.write(f"### Predicted Price : {result*1000:.2f} USD")
