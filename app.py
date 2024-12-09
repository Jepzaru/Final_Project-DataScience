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

st.set_page_config(
    page_title="Final Project",  
    page_icon="sjlogo.png",  
    layout="wide"  
)

DATA_PATH = r'train.csv'

if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
else:
    st.error(f"File not found at {DATA_PATH}. Please check the file path.")

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
    st.title('SuperStore Sales')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.image('salestore.jpg', use_container_width=True)
    st.title('Overview')
    st.markdown("""
    ### Retail Dataset of a Global Superstore (4 years)
    - **Dataset Description**: This dataset contains retail sales data from a global superstore over a span of 4 years.
    - **Research Question**: How can we predict the sales of the next 7 days from the last date of the training dataset?
    - **Selected Analysis Technique**: Time series analysis will be used to extract patterns and predict future sales. Time series analysis is ideal for non-stationary data like retail sales, weather data, stock prices, etc.
    - **Dataset Structure**:
        - Columns: `OrderID`, `OrderDate`, `ShipDate`, `ShipMode`, `CustomerID`, `Segment`, `Country`, `City`, `Category`, `Sub-Category`, `ProductName`, `Sales`, `Quantity`, `Discount`, `Profit`.
        - Description: The dataset captures order details, shipment information, product sales, and financial metrics like discounts and profit.
    """)


    st.subheader('Dataset Structure')
    df = pd.read_csv('train.csv')
    st.dataframe(df.head())
    
elif current_page == 'introduction':
    st.title('Welcome to WeatherWeatherLang')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.image('cube.gif', width=300)
    st.title('Introduction')
    st.markdown("""
    ### Rain Prediction
    Predicting rain for the next day using machine learning can provide numerous benefits. 
    It enables more informed decision-making for individuals and businesses, such as planning outdoor activities, 
    scheduling events, and optimizing agricultural practices. Accurate rain predictions can also help 
    utilities manage water resources more efficiently, reduce flooding risks, and improve disaster preparedness. 
    For industries reliant on weather conditions, like transport and construction, it allows for better planning 
    and resource allocation, minimizing delays and costs. Moreover, it can enhance personal safety by alerting people 
    about adverse weather conditions and encouraging them to take precautions. Overall, using machine learning to predict 
    rain can lead to a more efficient, safer, and proactive approach to managing weather-dependent activities.
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
        st.image('rays.png', width=300) 
        st.caption('Rise Jade R. Benavente')

    with col4:
        st.image('saly.png', width=300)  
        st.caption('Thesaly T. Tejano')

    with col5:
        st.image('shana.jpg', width=300) 
        st.caption('Xianna Andrei Cabana')
        
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
    3. Sales Distribution
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

    # Sales Distribution with Plotly
    st.subheader('3. Sales Distribution')

    # Dynamic control for number of bins in histogram
    num_bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=30)
    
    # Plotly histogram for sales distribution
    fig = px.histogram(df_clean, x='Sales', nbins=num_bins, title="Sales Distribution")
    fig.update_layout(xaxis_title='Sales', yaxis_title='Frequency')
    st.plotly_chart(fig)

    # Clustering with Plotly
    st.subheader('4. Clustering (Customer Segmentation)')
    st.write("K-Means Clustering on Sales and Category")

    if 'Sales' in df_clean.columns and 'Category' in df_clean.columns:
        # One-hot encode the 'Category' column
        category_dummies = pd.get_dummies(df_clean['Category'], prefix='Category')
        cluster_data = pd.concat([df_clean[['Sales']], category_dummies], axis=1)

        # Dynamic control for the number of clusters in K-Means
        num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(cluster_data)
        cluster_data['Cluster'] = kmeans.labels_

        # Interactive scatter plot with Plotly
        fig = px.scatter(cluster_data, x='Sales', y=category_dummies.columns[0], color='Cluster', title=f'K-Means Clustering with {num_clusters} Clusters')
        st.plotly_chart(fig)
    else:
        st.error("'Sales' or 'Category' column is missing in the data!")

    # Linear Regression with Plotly
    st.subheader('5. Linear Regression (Sales Prediction)')
    st.write("Using 'Category' to predict 'Sales'")

    # Check if necessary columns exist
    required_columns = ['Category', 'Sales']
    if all(col in df_clean.columns for col in required_columns):
        # One-hot encode the 'Category' column
        category_dummies = pd.get_dummies(df_clean['Category'], prefix='Category')
        features = category_dummies  # Use encoded 'Category' columns as features
        target = df_clean['Sales']
        
        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Fit the Linear Regression model
        model = LinearRegression().fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        st.write(f"Model Coefficients: {model.coef_}")
        st.write(f"Mean Squared Error: {mse}")

        # Plot Actual vs Predicted Sales using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
        fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Actual Sales', yaxis_title='Predicted Sales')
        st.plotly_chart(fig)
    else:
        st.error("Required columns for Linear Regression are missing!")

elif current_page == 'data_viz':
    st.title('Data Visualization')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    ### Data Visualization Page
    - Display charts and graphs.
    - Explore data interactively.
    - Understand key metrics visually.
    """)
