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
import numpy as np

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
        "introduction": "✨ Introduction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "overview": "📝 Overview &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "analytics": "⚡ Data Exploration&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "insights": "📈 Analysis and Insights&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   ",
         "conclusion": "📌 Conclusion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"  
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
    - *Dataset Description*: This dataset contains information about laptops and their prices.
    - *Research Question*: Can we predict the price of a laptop based on its features such as brand, processor, RAM, and storage?
    - *Dataset Structure*:
        - Columns: Company, TypeName, RAM, Weight, Price, TouchScreen, Ips, Ppi, Cpu_brand, HDD, SSD, Gpu_brand, Os.
        - Description: The dataset includes features such as company name, brand, processor type, RAM size, storage capacity, screen size, and the corresponding price of each laptop.
    """)
    
    # Allow users to select how many rows to display in the dataset structure
    st.subheader('Dataset Structure')
    st.dataframe(df.head(), use_container_width=True)

    # Clean column names
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.lower()  # Optional: Convert to lowercase for consistency

    with st.expander('▼ Want to customize what columns to see?'):
        num_rows = st.slider('Select number of rows to display', min_value=5, max_value=100, value=10)
        selected_columns = st.multiselect('Select columns to display', options=df.columns.tolist(), default=df.columns.tolist())
        st.write(df[selected_columns].head(num_rows), use_container_width=True)

    st.subheader('Summary Statistics')
    st.write(df.describe())

    # Clean column names
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    df.columns = df.columns.str.lower()  # Optional: Convert to lowercase for consistency

    # If RAM column is not numeric, extract numeric values
    if df['ram'].dtype == 'object':  # If 'RAM' column is a string
        df['ram'] = df['ram'].str.extract('(\d+)').astype(int)

    # Filter only numerical columns for correlation
    numerical_df = df.select_dtypes(include=[np.number])

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

    num_bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=30)
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


elif current_page == 'insights':
    st.title('Analysis and Insights')
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.subheader('1. Key Insights from Data Exploration and Clustering')
    st.write("""
        - **Laptop Price Segmentation**: From the K-Means clustering, we observed distinct segments based on price and storage. These segments help identify high-end laptops with larger storage capacities, as well as budget laptops with more modest specifications.
        - **Price vs. Storage**: The clustering model showed a correlation between storage capacity and price, with laptops having larger storage being more expensive.
        - **Predicted Price Trends**: The linear regression model revealed that both processor type and RAM have a significant impact on predicting the laptop price, with higher-end processors and more RAM correlating with higher prices.
    """)

    # Display the clustering insights
    st.subheader('2. Clustering Results')
    st.write("The following plot shows the results of K-Means clustering based on laptop prices and storage capacity.")
    
    # Recreate the plot from earlier if clustering data exists
    if 'Cluster' in cluster_data.columns:
        fig = px.scatter(cluster_data, x='Price', y=storage_dummies.columns[0], color='Cluster', 
                         title='Laptop Price Segmentation via K-Means Clustering')
        st.plotly_chart(fig)
    
    st.write("""
        - The clustering analysis provides insights into the distribution of laptops based on their price and storage.
        - The clusters show that laptops with larger storage tend to have a higher price, which aligns with expectations.
    """)

    # Display insights from the linear regression model
    st.subheader('3. Linear Regression Results')
    st.write("""
        - **Model Performance**: The linear regression model predicted laptop prices with a mean squared error of `X` (replace X with your computed value), suggesting a relatively good fit for the data.
        - **Coefficient Interpretation**: Higher-end processors and larger amounts of RAM are significant predictors of laptop price, with each additional GB of RAM and processor upgrade contributing positively to the price.
    """)

    st.write("Below is a scatter plot showing the actual vs. predicted prices using the linear regression model.")
    
    # Scatter plot for actual vs predicted prices
    if 'y_test' in locals() and 'predictions' in locals():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
        fig.update_layout(title='Actual vs Predicted Price', xaxis_title='Actual Price', yaxis_title='Predicted Price')
        st.plotly_chart(fig)

    # Conclusion and insights
    st.subheader('4. Key Takeaways and Recommendations')
    st.write("""
        - **Laptop Price Prediction**: The analysis highlights how various features, such as processor, RAM, and storage, influence laptop prices. This information can guide consumers in selecting laptops within their budget based on their requirements.
        - **Future Work**: The model could be improved further by adding more features like brand, screen size, and GPU type to improve prediction accuracy.
        - **Business Use**: Businesses selling laptops can use this model to price their laptops more competitively by understanding which features are most valued by consumers.
    """)

elif current_page == 'conclusion':
    st.title('Conclusion and Future Work')
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.subheader("Conclusion")
    st.write("""
    - This project demonstrates the ability to predict laptop prices using various machine learning techniques, including K-Means clustering and Linear Regression.
    - The insights gained from the data exploration and analysis can help businesses and consumers make informed decisions.
    - The model performed well in predicting laptop prices, although there is potential for improvement by incorporating more features and tuning the model.
    """)

    st.subheader("Main Takeaways and Recommendations")
    st.write("""
    - **Main Takeaways**:
        - Machine learning models such as Linear Regression can be effectively used for price prediction tasks.
        - Data exploration revealed key factors that influence laptop pricing, such as brand, specifications, and market demand.
        - While the model shows promising results, there is room for improvement through feature engineering and model tuning.
    
    - **Actionable Recommendations**:
        - **Businesses**: Leverage the insights from the data analysis to optimize inventory and pricing strategies.
        - **Consumers**: Use the predictions to compare prices and make informed purchasing decisions.
        - **Future Improvements**: Explore advanced machine learning models, and consider expanding the dataset for better accuracy.
    """)

    st.subheader("Future Work")
    st.write("""
    - **Enhanced Data Collection**: Future work could involve collecting more data, particularly for missing values, which could improve the accuracy of the models.
    - **Model Improvement**: We could explore other machine learning models, such as Random Forest or XGBoost, for more accurate price predictions.
    - **Interactive Dashboard**: The development of an interactive dashboard for end-users to input laptop features and receive price predictions could be a valuable extension of this project.
    """)

    st.subheader("Final Thoughts")
    st.write("""
    In conclusion, this project provides a strong foundation for laptop price prediction through data-driven approaches. By leveraging machine learning models and insightful data analysis, we successfully demonstrated the practical application of predictive analytics. 
    Moving forward, the proposed enhancements could elevate the utility and precision of this project, ultimately contributing to its real-world applicability for both consumers and businesses.
    """)

    st.subheader("Explore Insights")
    st.write("Use the options below to interact with the insights:")

    # Dropdown or text box to explore further insights
    option = st.selectbox(
        'Select an insight to explore further:',
        ['Price Trends by Brand', 'Price Distribution by Specs', 'Best Models for Investment']
    )

    if option == 'Price Trends by Brand':
        st.write("Here, you can explore how different brands influence the laptop price trends.")
        
        # Visualize price trends by Company (Brand)
        brand_price_trend = df.groupby('Company')['Price'].mean().sort_values(ascending=False)
        st.write("Average Price by Company:")
        st.bar_chart(brand_price_trend)

        st.write("""
        - From the chart, we can observe how different companies perform in terms of average price. Higher-end companies tend to have higher average prices, while budget-friendly companies tend to have more affordable options.
        - Insights like these can help both consumers and businesses make informed decisions regarding brand preferences based on their pricing strategy or purchasing budget.
        """)

        # Plotly interactive boxplot for price distribution by company
        fig = px.box(df, 
                    x='Company', 
                    y='Price', 
                    title='Price Distribution by Company',
                    labels={'Price': 'Laptop Price', 'Company': 'Brand'},
                    points='all'  # Show all points in the boxplot for clarity
                    )
        
        fig.update_layout(
            xaxis_tickangle=45  # Rotate the x-axis labels for better visibility
        )

        st.plotly_chart(fig)

    elif option == 'Price Distribution by Specs':
        st.write("Explore the price distribution across different laptop specifications.")
        
        # Price distribution by RAM size
        ram_price_dist = df.groupby('Ram')['Price'].mean().reset_index()
        st.write("Price Distribution by RAM Size:")
        fig_ram = px.bar(ram_price_dist, x='Ram', y='Price', title='Price Distribution by RAM Size',
                        labels={'Ram': 'RAM Size (GB)', 'Price': 'Average Price'},
                        color='Price', color_continuous_scale='Viridis')
        st.plotly_chart(fig_ram)

        # Price distribution by Storage type (HDD and SSD)
        # Assuming HDD and SSD are separate columns, we can visualize their impact on price
        hdd_price_dist = df.groupby('HDD')['Price'].mean().reset_index()
        ssd_price_dist = df.groupby('SSD')['Price'].mean().reset_index()

        st.write("Price Distribution by HDD Size:")
        fig_hdd = px.bar(hdd_price_dist, x='HDD', y='Price', title='Price Distribution by HDD Size',
                        labels={'HDD': 'HDD Size (GB)', 'Price': 'Average Price'},
                        color='Price', color_continuous_scale='Viridis')
        st.plotly_chart(fig_hdd)
        
        st.write("Price Distribution by SSD Size:")
        fig_ssd = px.bar(ssd_price_dist, x='SSD', y='Price', title='Price Distribution by SSD Size',
                        labels={'SSD': 'SSD Size (GB)', 'Price': 'Average Price'},
                        color='Price', color_continuous_scale='Viridis')
        st.plotly_chart(fig_ssd)

        st.write("""
        - We can observe trends based on specifications like RAM, HDD, and SSD. Laptops with higher RAM and storage configurations tend to have higher prices.
        - Understanding these trends can help consumers choose the specifications that fit their budget, while businesses can use this data for inventory management and pricing strategies.
        """)

        # Additional visualization: Scatter plot to analyze the relationship between price and RAM/storage
        # Price vs RAM
        fig_ram_scatter = px.scatter(df, x='Ram', y='Price', title='Price vs RAM Size',
                                    labels={'Ram': 'RAM Size (GB)', 'Price': 'Laptop Price'},
                                    color='Price', color_continuous_scale='Viridis')
        st.plotly_chart(fig_ram_scatter)

        # Price vs SSD
        fig_ssd_scatter = px.scatter(df, x='SSD', y='Price', title='Price vs SSD Size',
                                    labels={'SSD': 'SSD Size (GB)', 'Price': 'Laptop Price'},
                                    color='Price', color_continuous_scale='Viridis')
        st.plotly_chart(fig_ssd_scatter)

    elif option == 'Best Models for Investment':
        st.write("Find the best models based on value for money.")
        
        # Calculate value for money using Price and RAM as an indicator (you can adjust this based on your dataset's features)
        df['value_for_money'] = df['Price'] / df['Ram']  # Simplified assumption: price-to-RAM ratio
        top_investment_models = df[['Company', 'TypeName', 'value_for_money']].sort_values(by='value_for_money').head(10)
        
        st.write("Top 10 Models Based on Value for Money:")
        st.dataframe(top_investment_models)

        st.write("""
        - These models are ranked based on their price-to-RAM ratio. Lower values indicate better value for money, making these models more desirable for cost-conscious consumers.
        - Businesses may also consider this analysis when selecting models to include in their inventory based on performance relative to cost.
        """)

        # Plot top 10 models based on value for money using Plotly
        fig = px.bar(top_investment_models, 
                    x='value_for_money', 
                    y='TypeName', 
                    color='Company', 
                    orientation='h',  # Horizontal bar chart
                    title='Top 10 Models Based on Value for Money',
                    labels={'value_for_money': 'Price to RAM Ratio', 'TypeName': 'Laptop Model'},
                    hover_data=['Company', 'value_for_money'])
        
        st.plotly_chart(fig)
