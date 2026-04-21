
import streamlit as st
import pandas as pd
import pickle as pkl

# Load the trained model
@st.cache_resource # Cache the model loading to prevent re-loading on each rerun
def load_model():
    try:
        with open('gradient_boosting_df1_model.pkl', 'rb') as file:
            model = pkl.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'gradient_boosting_df1_model.pkl' not found. Please ensure it's in the same directory.")
        st.stop()

model = load_model()

st.title('Salary Prediction App')
st.write('Enter the details to predict the salary in USD.')

# Input fields for the features
work_year = st.slider('Work Year', min_value=2020, max_value=2025, value=2022)
remote_ratio = st.slider('Remote Ratio (0 = On-site, 50 = Hybrid, 100 = Remote)', min_value=0, max_value=100, step=50, value=100)

company_size_map = {'Small': 2, 'Medium': 1, 'Large': 0}
company_size_display = st.selectbox('Company Size', options=list(company_size_map.keys()))
company_size_encoded = company_size_map[company_size_display]

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'work_year': [work_year],
    'remote_ratio': [remote_ratio],
    'company_size_encoded': [company_size_encoded]
})

if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f} USD')
```
