import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go  # Import Plotly Graph Objects
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from colorama import Fore, Back, Style
import plotly.express as px


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
        "analytics": "⚡ Data Exploration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "analy_ins": "📈 Analysis and Insights &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "data_viz": "📌 Conclusion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
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
    # Create three columns: one for left, one for center, and one for right
    col1, col2, col3 = st.columns([2, 2, 1])  # Adjust the weight to center the image
    
    # Place the image in the center column
    with col2:
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
    4. Clustering: Similar Laptop Classification
    5. Linear Regression: Modeling Price
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

    # Clustering
    st.subheader('4. Clustering: Similar Laptop Classification')
    st.write("🔵 K-means Clustering: Partitioning data into distinct clusters")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Select numeric columns for clustering
    numeric_cols = df_clean.select_dtypes(include='number')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_cols)

    # Elbow method to find optimal number of clusters
    loss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        loss.append(kmeans.inertia_)

    # Plot Elbow Curve
    fig = px.line(x=list(range(1, 11)), y=loss, markers=True, title="Elbow Method for Optimal Clusters")
    fig.update_layout(xaxis_title='Number of Clusters (k)', yaxis_title='Inertia (Loss)')
    st.plotly_chart(fig)

    # K-means with 5 clusters
    k = st.number_input("Select Number of Clusters for K-means", min_value=2, max_value=10, value=5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    df_clean['Cluster'] = labels_kmeans
    st.write("Clustered Data:")
    st.dataframe(df_clean)

    # Linear Regression
    st.subheader('5. Linear Regression: Modeling Price')
    st.write("📈 Linear Regression: Modeling the relationship between features and price")

    # Ensure required columns for linear regression
    required_columns = ['Cpu_brand', 'Ram', 'Price']

    # Check and log available columns in df_clean
    st.write("Available columns in the cleaned dataset:")
    st.write(df_clean.columns)

    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        st.success("All required columns for Linear Regression are present!")
        
        # Check for missing data in the required columns
        st.write("Checking for missing values in the required columns...")
        st.write(df_clean[required_columns].isnull().sum())

        # Proceed only if no missing data in the required columns
        if df_clean[required_columns].isnull().sum().sum() > 0:
            st.error("There are missing values in the required columns. Please handle them before proceeding!")
        else:
            # One-hot encoding
            processor_dummies = pd.get_dummies(df_clean['Cpu_brand'], prefix='Cpu_brand')
            ram_dummies = pd.get_dummies(df_clean['Ram'], prefix='Ram')
            features = pd.concat([processor_dummies, ram_dummies], axis=1)
            target = df_clean['Price']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Fit model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            mse_lr = mean_squared_error(y_test, y_pred)

            st.write("### Model Results")
            st.write(f"Model Coefficients: {lr.coef_}")
            st.write(f"Mean Squared Error: {mse_lr:.2f}")

            # Plot Actual vs Predicted
            st.write("### Actual vs Predicted Prices")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
            fig.update_layout(
                title='Actual vs Predicted Price',
                xaxis_title='Actual Price',
                yaxis_title='Predicted Price',
                template='plotly_white'
            )
            st.plotly_chart(fig)

elif current_page == 'analy_ins':
    st.title('Analysis and Insights')

    # Add data visualization-related content here
    st.markdown('### Analysis and Insights')


elif current_page == 'data_viz':
    st.title('Conclusion')

    # Add data visualization-related content here
    st.markdown('### Placeholder for Conlcusion')

