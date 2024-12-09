import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(
    page_title="Final Project",  
    page_icon="sjlogo.png",  
    layout="wide"  
)

# Update the data path to 'laptop.csv'
DATA_PATH = r'laptop.csv'

# Load dataset
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
else:
    st.error(f"File not found at {DATA_PATH}. Please check the file path.")

# Sidebar function
def sidebar():
    # Get the current page from the session state or default to 'overview'
    current_page = st.session_state.get('page', 'overview')

    # Sidebar content
    st.sidebar.image('sjlogo.png', width=400)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # Define navigation links and their corresponding page keys
    nav_links = {
        "introduction": "✨ Introduction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "overview": "📝 Overview &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "analytics": "⚡ Data Exploration&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "data_viz": "📈 Data Visualization&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
    }

    # Create a container for buttons and render navigation links
    st.sidebar.markdown('<div class="sidebar-button-container">', unsafe_allow_html=True)
    for key, label in nav_links.items():
        active_class = 'active' if current_page == key else ''
        if st.sidebar.button(label, key=key):
            st.session_state['page'] = key
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Initialize the page state
if 'page' not in st.session_state:
    st.session_state['page'] = 'introduction'

# Render the sidebar
sidebar()

# Main content area
current_page = st.session_state['page']

if current_page == 'overview':
    st.title('Laptop Price Prediction')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.image('laptop.jpg', use_container_width=True)
    st.title('Overview')
    st.markdown("""
    ### Laptop Price Dataset
    - **Dataset Description**: This dataset contains information about laptops and their prices.
    - **Research Question**: Can we predict the price of a laptop based on its features such as brand, processor, RAM, and storage?
    - **Dataset Structure**:
        - Columns: `Brand`, `Processor`, `RAM`, `Storage`, `Screen Size`, `Price`.
        - Description: The dataset includes features like the brand, processor type, RAM size, storage capacity, screen size, and the corresponding price of each laptop.
    """)
    
    st.subheader('Dataset Structure')
    st.dataframe(df.head())

elif current_page == 'introduction':
    st.title('Welcome to Laptop Price Prediction')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.image('cube.gif', width=300)
    st.title('Introduction')
    st.markdown("""
    ### Predicting Laptop Prices
    Predicting laptop prices using machine learning can assist consumers in making informed purchasing decisions, 
    and it can help businesses set competitive prices. By using various features like processor type, RAM, 
    storage, and screen size, we can build a model that estimates laptop prices.
    """)

    st.title('Team Members')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image('Jep.png', width=300)  
        st.caption('Jeff Francis D. Conson')

    with col2:
        st.image('kem.png', width=300) 
        st.caption('Kimverly B. Bacalso')

    with col3:
        st.image('shana.jpg', width=300) 
        st.caption('Xianna Andrei Cabana')
        
    with col4:
        st.image('saly.png', width=300)  
        st.caption('Thesaly T. Tejano')

    with col5:
        st.image('rays.png', width=300) 
        st.caption('Rise Jade R. Benavente')
        
    st.markdown(
        """
        <style>
        .stImage img {
            height: 300px !important;
            border-radius: 15px;
        }
        .stCaption{
            align-text: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif current_page == 'analytics':
    st.title('Data Exploration and Analysis')
    st.markdown('<hr>', unsafe_allow_html=True)

    st.markdown("""
    ### Steps Included:
    1. Data Cleaning and Exploration
    2. Descriptive Statistics
    3. Laptop Price Distribution
    4. Clustering
    5. Linear Regression
    """)

    # Data Cleaning and Exploration
    st.subheader('1. Data Cleaning and Exploration')
    st.write("### Initial Data")
    df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
    st.dataframe(df)

    st.write("### Handling Missing Values")
    df_clean = df.dropna()  # Drop rows with missing values
    st.write("Cleaned Data (No Missing Values):")
    st.dataframe(df_clean)

    # Descriptive Statistics
    st.subheader('2. Descriptive Statistics')
    st.write(df_clean.describe())

    # Laptop Price Distribution with Plotly
    st.subheader('3. Laptop Price Distribution')

    # Dynamic control for number of bins in histogram
    num_bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=30)
    
    # Plotly histogram for laptop price distribution
    fig = px.histogram(df_clean, x='Price', nbins=num_bins, title="Laptop Price Distribution")
    fig.update_layout(xaxis_title='Price', yaxis_title='Frequency')
    st.plotly_chart(fig)

    # Clustering with Plotly
    st.subheader('4. Clustering (Laptop Segmentation)')
    st.write("K-Means Clustering on Price and Storage")

    # Check if 'Storage' column exists and handle missing columns
    if 'Storage' in df_clean.columns:
        # One-hot encode the 'Storage' column
        storage_dummies = pd.get_dummies(df_clean['Storage'], prefix='Storage')
        cluster_data = pd.concat([df_clean[['Price']], storage_dummies], axis=1)

        # Dynamic control for the number of clusters in K-Means
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(cluster_data)
        cluster_data['Cluster'] = kmeans.labels_

        # Interactive scatter plot with Plotly
        fig = px.scatter(cluster_data, x='Price', y=storage_dummies.columns[0], color='Cluster', title=f'K-Means Clustering with {num_clusters} Clusters')
        st.plotly_chart(fig)
    else:
        st.error("'Storage' column not found in the dataset!")

    # Linear Regression with Plotly
    st.subheader('5. Linear Regression (Price Prediction)')
    st.write("Using 'Processor' and 'RAM' to predict 'Price'")

    # Check if necessary columns exist for Linear Regression
    required_columns = ['Processor', 'RAM', 'Price']
    if all(col in df_clean.columns for col in required_columns):
        # One-hot encode the 'Processor' and 'RAM' columns
        processor_dummies = pd.get_dummies(df_clean['Processor'], prefix='Processor')
        ram_dummies = pd.get_dummies(df_clean['RAM'], prefix='RAM')
        features = pd.concat([processor_dummies, ram_dummies], axis=1)  # Use encoded columns as features
        target = df_clean['Price']
        
        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Fit the Linear Regression model
        model = LinearRegression().fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        st.write(f"Model Coefficients: {model.coef_}")
        st.write(f"Mean Squared Error: {mse}")

        # Plot Actual vs Predicted Price using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
        fig.update_layout(title='Actual vs Predicted Price', xaxis_title='Actual Price', yaxis_title='Predicted Price')
        st.plotly_chart(fig)
    else:
        st.error("Required columns for Linear Regression ('Processor', 'RAM', 'Price') not found!")

elif current_page == 'data_viz':
    st.title('Data Visualization')

    # Add data visualization-related content here
    st.markdown('### Placeholder for Data Visualization')

