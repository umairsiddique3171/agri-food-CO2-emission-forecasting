import os
import sys
sys.path.append(os.getcwd())

import streamlit as st

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import set_background, load_selected_features

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title='agri-food emissions',
    page_icon='🌾',
    layout='wide'
    )

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



set_background(os.path.join(os.getcwd(),'app','background_img.jpg'))

st.markdown("<h1 style='text-align:center'>Agri Food Emissions Prediction</h1>",unsafe_allow_html=True)

area_categories = load_selected_features(os.path.join('notebook','area_categories.json'))

features = load_selected_features(os.path.join('notebook','selected_features.json'))

form = st.form("Form 1")

col1,col2,col3,col4 = form.columns(4)

with col1:
    a = st.selectbox("Select Files",options=area_categories)
with col2: 
    b = st.slider(features[1],min_value=1980,max_value=2050,step=2)
with col3: 
    c = st.number_input(features[2],value=15)
with col4: 
    d = st.number_input(features[3],value=0.05)
with col1: 
    e = st.number_input(features[4],value=205)
with col2: 
    f = st.number_input(features[5],value=685)
with col3: 
    g = st.number_input(features[6],value=5)
with col4: 
    h = st.number_input(features[7],value=12)
with col1: 
    i = st.number_input(features[8],value=65)
with col2: 
    j = st.number_input(features[9],value=-2390)
with col3: 
    k = st.number_input(features[10],value=3)
with col4: 
    l = st.number_input(features[11],value=80)
with col1: 
    m = st.number_input(features[12],value=110)
with col2: 
    n = st.number_input(features[13],value=14)
with col3: 
    o = st.number_input(features[14],value=70)
with col4: 
    p = st.number_input(features[15],value=700)
with col1: 
    q = st.number_input(features[16],value=252)
with col2: 
    r = st.number_input(features[17],value=12)
with col3: 
    s = st.number_input(features[18],value=260)
with col4: 
    t = st.number_input(features[19],value=1590)
with col1: 
    u = st.number_input(features[20],value=9)
with col2: 
    v = st.number_input(features[21],value=4)
with col3: 
    w = st.number_input(features[22],value=11)
with col4: 
    x = st.number_input(features[23],value=10000000)
with col1: 
    y = st.number_input(features[24],value=2500000)

btn = form.form_submit_button("Predict")

if btn : 
    input_list = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y]
    if 0 in input_list or "" in input_list: 
        missing_values = [i for i, val in enumerate(input_list) if val == 0 or val == '']
        missing_values = [features[i] for i in missing_values]
        st.warning(f"{', '.join(missing_values)} (missing)")
    else : 
        st.success("Submitted Successfully!")
        input_data = CustomData(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y)
        predictor = PredictPipeline()
        prediction = predictor.predict(input_data=input_data.get_data_as_data_frame())
        st.markdown("### Results")
        st.markdown(f"``Total Emission : {prediction:.3f}``")
