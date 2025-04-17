import streamlit as st
import pandas as pd
import plotly.express as px
from zomato_predictor import predict_price, load_data
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Zomato Price Predictor",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🍽️ Restaurant Price Predictor")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application predicts restaurant prices using machine learning.
    The model is trained on Zomato restaurant data and considers various
    factors to make accurate predictions.
    """)
    st.markdown("---")
    st.subheader("How to use:")
    st.write("""
    1. Enter restaurant details
    2. Click 'Predict Price'
    3. View the prediction and insights
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Data Insights")
    
    # Load and display sample data
    df = load_data()
    st.write("Sample Data Overview:")
    st.dataframe(df.head(5))
    
    # Add visualizations
    fig1 = px.scatter(df, 
                      x="rating", 
                      y="average_cost_for_two",
                      title="Rating vs Cost",
                      color="has_online_delivery")
    st.plotly_chart(fig1)

with col2:
    st.subheader("🎯 Price Prediction")
    with st.form("prediction_form"):
        votes = st.number_input("Number of Votes", min_value=0, max_value=10000, value=100)
        rating = st.slider("Rating", 0.0, 5.0, 3.5)
        has_table_booking = st.selectbox("Table Booking Available?", ["Yes", "No"])
        has_online_delivery = st.selectbox("Online Delivery Available?", ["Yes", "No"])
        
        submit = st.form_submit_button("Predict Price")
        
        if submit:
            # Prepare input data
            input_data = pd.DataFrame({
                'votes': [votes],
                'rating': [rating],
                'has_table_booking': [1 if has_table_booking == "Yes" else 0],
                'has_online_delivery': [1 if has_online_delivery == "Yes" else 0]
            })
            
            # Get prediction
            predicted_price = predict_price(input_data)
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                    <h3 style='color: #ff4b4b; text-align: center;'>
                        Predicted Price: ₹{predicted_price:.2f}
                    </h3>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
