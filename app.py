import streamlit as st
import numpy as np
from prediction import get_yield
import joblib 

model=joblib.load('blueberry.sav')

st.set_page_config(page_title='Wild Blueberry Yield Prediction App',page_icon='🫐',layout='wide')

st.markdown("<h1 style = 'text-align: center;color: #8A03DD;'>Wild Blueberry Yield Prediction 🫐</h1>", unsafe_allow_html=True)
st.image('blue.jpg',use_column_width=True)

def main():
    with st.form('prediction form'):
        st.subheader("Enter the values for the following features")

        clonesize= st.text_input('Enter the clone size')
        honeybee=st.text_input('Enter the density of honey bees')
        bumbles=st.text_input('Enter the density of bumble bees')
        andrena=st.text_input('Enter the density of andrena bees')
        osmia=st.text_input('Enter the density of osmia bees')
        MaxOfUpperTRange=st.text_input('Enter the maximum temperature attained during the blooming season')
        RainingDays= st.text_input('Enter the number of days of rain in the blooming season')
        submit=st.form_submit_button('Predict Yield')

    if submit:
        clonesize=float(clonesize)
        honeybee=float(honeybee)
        bumbles=float(bumbles)
        andrena=float(andrena)
        osmia=float(osmia)
        MaxOfUpperTRange=float(MaxOfUpperTRange)
        RainingDays=float(RainingDays)
        data=np.array([clonesize,honeybee,bumbles,andrena,osmia,MaxOfUpperTRange,RainingDays]).reshape(1,-1)
        pred=get_yield(model=model,data=data)
        res=round(pred[0],2)
        st.write('Predicted yield of Wild Blueberry is {} kg/hectare'.format(res))
    
if __name__=='__main__':
    main()