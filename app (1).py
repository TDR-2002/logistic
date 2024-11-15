import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stForm {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
    This app predicts the probability of survival for a Titanic passenger based on their characteristics.
    Enter the passenger details below to get a prediction.
""")

# Sidebar for additional information
with st.sidebar:
    st.header("About")
    st.info("""
        This model is trained on the Titanic dataset using Logistic Regression.
        It considers various factors such as:
        - Passenger Class
        - Gender
        - Age
        - Family Size
        - Ticket Fare
        - Port of Embarkation
    """)
    
    st.header("Model Performance")
    st.success("""
        - Accuracy: 82%
        - Precision: 78%
        - Recall: 71%
        - F1-Score: 74%
    """)

# Main form
with st.form("prediction_form", clear_on_submit=False):
    st.subheader("Passenger Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            help="1 = 1st class (Upper), 2 = 2nd class (Middle), 3 = 3rd class (Lower)"
        )
        
        sex = st.selectbox(
            "Gender",
            options=["male", "female"]
        )
        
        age = st.slider(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            help="Age of the passenger"
        )
        
        fare = st.slider(
            "Ticket Fare (£)",
            min_value=0,
            max_value=600,
            value=32,
            help="Price paid for the ticket"
        )
    
    with col2:
        sibsp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of siblings or spouses traveling with the passenger"
        )
        
        parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of parents or children traveling with the passenger"
        )
        
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["C", "Q", "S"],
            help="C = Cherbourg, Q = Queenstown, S = Southampton"
        )
        
        total_family = sibsp + parch
        st.info(f"Total family members aboard: {total_family}")

    submit_button = st.form_submit_button(
        "Predict Survival",
        help="Click to get survival prediction"
    )

# Function to prepare input data
def prepare_input_data(pclass, sex, age, sibsp, parch, fare, embarked):
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Encode categorical variables
    sex_encoder = LabelEncoder()
    embarked_encoder = LabelEncoder()
    
    sex_encoder.fit(['male', 'female'])
    embarked_encoder.fit(['C', 'Q', 'S'])
    
    input_data['Sex'] = sex_encoder.transform(input_data['Sex'])
    input_data['Embarked'] = embarked_encoder.transform(input_data['Embarked'])
    
    return input_data

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('titanic_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is saved correctly.")
        return None

model = load_model()

# Make prediction when form is submitted
if submit_button:
    if model is not None:
        # Prepare input data
        input_data = prepare_input_data(pclass, sex, age, sibsp, parch, fare, embarked)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Create columns for results
        st.markdown("---")
        st.header("Prediction Results")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            if prediction == 1:
                st.success("#### Passenger Would Likely SURVIVE")
            else:
                st.error("#### Passenger Would Likely NOT SURVIVE")
        
        with col2:
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Survival Probability"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.info("""
            #### Key Survival Factors:
            - Being female greatly increased survival chances
            - First-class passengers had better odds
            - Children were prioritized
            - Small families had higher survival rates
            """)
        
        # Additional analysis
        st.markdown("---")
        st.subheader("Similar Historical Cases")
        
        historical_context = f"""
        Based on historical data from the Titanic:
        - {sex.capitalize()} passengers in {pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} class had a 
          {'higher' if (sex=='female' or pclass==1) else 'lower'} survival rate
        - Passengers of age {age} were {'more likely' if age < 15 or age > 60 else 'less likely'} to survive
        - Those who embarked from {{'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}[embarked]} had 
          {'better' if embarked=='C' else 'average' if embarked=='Q' else 'lower'} survival rates
        """
        st.write(historical_context)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Based on Titanic dataset • Model: Logistic Regression</p>
</div>
""", unsafe_allow_html=True)