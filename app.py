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
    - *Selected Analysis Technique*: 
        - K-means Clustering: Helps by identifying distinct segments or clusters of laptops with similar features, such as brand, specifications, and price ranges. By grouping laptops into different clusters, the model can identify trends and patterns specific to each segment, allowing for more accurate price prediction. For example, laptops in a high-performance cluster may have different pricing factors compared to those in a budget-friendly cluster.
        - Linear Regression: Helps predict the price of a laptop by finding the best-fit line that describes how the price changes with respect to the various features. This technique is essential for understanding the impact of specific features on the price and providing a quantitative way to forecast future prices based on these attributes. It is especially useful when the relationship between the variables is linear or approximately linear.
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

    with st.expander('▼ Customize what columns to display'):
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

    # Data Correlation (Dropping object columns)
    # Select only numeric columns (int64, float64)
    numeric_df = df_clean.select_dtypes(include=[np.number])

    # Calculate correlation on numeric columns only
    corr_matrix = numeric_df.corr()

    # Display the correlation matrix

    # Correlation Heatmap using Plotly
    st.subheader('5. Correlation Heatmap')
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Viridis', title="Correlation Heatmap")
    st.plotly_chart(fig)

    # Scatter plot of RAM vs Price
    st.subheader('6. Scatter Plot: RAM vs Price')
    fig = px.scatter(df_clean, x='Ram', y='Price', labels={'Ram': 'RAM', 'Price': 'Price'})
    st.plotly_chart(fig)

    # Bar plot of Company vs Price
    st.subheader('7. Bar Plot: Company vs Price')
    bar_data = df_clean.groupby('Company')['Price'].mean().reset_index()
    st.bar_chart(bar_data.set_index('Company'))

    # Word Cloud for Company, Gpu_brand, and Os
    st.subheader('8. Word Cloud for Company, Gpu_brand, and Os')
    text = ' '.join(df_clean["Company"].values.tolist() + df_clean["Gpu_brand"].values.tolist() + df_clean["Os"].values.tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))  # Set the figure size
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Data Cleaning and Exploration
    st.subheader('Data Cleaning and Data Preprocessing')
    df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
    st.write("#### Cleaned Data (No Missing Values):")
    # Check for missing values and display the result as a DataFrame
    missing_values = df.isnull().sum()
    st.dataframe(missing_values.to_frame(name='Null Count'))
    # Display Jupyter Code for Feature Engineering Process

    st.write('#### Feature Engineering Process')

    # Feature extraction
    df['ScreenSize'] = df['Ppi'] * df['Weight']
    df['StorageCapacity'] = df['HDD'] + df['SSD']
    st.dataframe(df[['ScreenSize', 'StorageCapacity']].head())
    st.write("#### Encoding Categorical Features")
    df_encoded = pd.get_dummies(df, columns=['Company', 'TypeName', 'TouchScreen', 'Cpu_brand', 'Gpu_brand', 'Os'])
    st.dataframe(df_encoded.head())

    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    # Splitting the data into train and test sets
    st.write("#### Splitting the Data into Train and Test Sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

    

    # Skip Feature Scaling section (No Scaling)
    # If you don't want to scale, just remove this section.
    # The X_train and X_test will not be scaled.

    # Label Encoding for Target Variable
    st.write("#### Label Encoding for Target Variable")
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    st.write("Encoded Target Variable (First 5 rows of y_train):")
    st.dataframe(y_train[:5])

    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (Linear Regression): {mse_lr:.2f}")
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['X'] = X



elif current_page == 'insights':
    st.title('Analysis and Insights')
    st.markdown('<hr>', unsafe_allow_html=True)
    
    if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test', 'X']):
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        X = st.session_state['X']

        st.write(f"Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

         # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Predict the values
        y_pred = lr.predict(X_test)

        # Calculate MSE
        mse_lr = mean_squared_error(y_test, y_pred)

        # Streamlit display for MSE
        st.write(f"**Mean Squared Error (Linear Regression):** {mse_lr:.2f}")

        # Visualization: Plot Actual vs Predicted Values using Streamlit
        st.write("#### Linear Regression: Actual vs Predicted")

        # Set black background style
        plt.style.use('dark_background')

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Scatter plot of actual vs predicted values
        plt.scatter(y_test, y_pred, color='red', label="Predicted vs Actual")

        plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='cyan', linestyle='--', linewidth=2, label="Perfect Prediction Line")
        print(f"y_test range: {y_test.min()} to {y_test.max()}")


        # Labels and title
        plt.title('Linear Regression: Actual vs Predicted Values', color='white')
        plt.xlabel('Actual Values', color='white')
        plt.ylabel('Predicted Values', color='white')
        plt.legend(facecolor='black', edgecolor='white', loc='upper left')
        plt.grid(True, color='white')

        # Display the plot dynamically in Streamlit
        st.pyplot(plt, use_container_width=False)
        
        st.write("""The chart is a scatter plot showing the relationship between the actual values (true values from the dataset) 
                 and the predicted values generated by a linear regression model. Each point on the scatter plot represents a single 
                 data point, with the actual value plotted on the x-axis and the corresponding predicted value on the y-axis. 
                 The closer the points are to the diagonal line, which represents a perfect prediction scenario where the predicted 
                 values exactly match the actual values, the better the model's performance. Points that are farther from the line indicate 
                 larger prediction errors, suggesting that the model does not perform as well for those data points. The dashed turquoise 
                 line serves as a reference to visualize how well the model aligns with the true values. This chart helps assess the accuracy 
                 of the linear regression model in predicting the target variable.""")

        # Elbow Method for KMeans Clustering
        loss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            loss.append(kmeans.inertia_)

        # Streamlit Dynamic Plotting
        st.title('KMeans Elbow Method')

        # Line Plot for Elbow Method
        st.write("#### Elbow Method for Optimal k")

        # Create a DataFrame for elbow method values
        elbow_df = pd.DataFrame({
            "Number of Clusters": range(1, 11),
            "Inertia": loss
        })

        # Use Streamlit-native line chart for the Elbow Method plot
        st.line_chart(elbow_df.set_index("Number of Clusters"))

        # KMeans Clustering with 5 clusters
        km = KMeans(n_clusters=5)
        km.fit_predict(X_train, y_train)  # Only use X_train for clustering
        labels_kmeans = km.labels_

        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X_train, y_train)  # Only use X_train for clustering

        # Silhouette Score for KMeans
        st.write("#### KMeans Clustering: Silhouette Score", silhouette_score(X_train, labels_kmeans))
        st.write("#### DBSCAN Clustering: Silhouette Score", silhouette_score(X_train, labels_dbscan))

        data = {
            'Clustering Algorithm': ['K-means', 'DBSCAN'],
            'Silhouette Score': [silhouette_score(X_train, labels_kmeans), silhouette_score(X_train, labels_dbscan)]
        }
        df = pd.DataFrame(data)

        # Display the DataFrame in Streamlit (optional)
        st.write(df)

        # Create the bar chart
        st.bar_chart(df.set_index('Clustering Algorithm')['Silhouette Score'])
        
        st.write("""The chart is a scatter plot showing the relationship between the actual values (true values from the dataset) 
                 and the predicted values generated by a linear regression model. Each point on the scatter plot represents a single 
                 data point, with the actual value plotted on the x-axis and the corresponding predicted value on the y-axis. 
                 The closer the points are to the diagonal line, which represents a perfect prediction scenario where the predicted 
                 values exactly match the actual values, the better the model's performance. Points that are farther from the line indicate 
                 larger prediction errors, suggesting that the model does not perform as well for those data points. The dashed turquoise 
                 line serves as a reference to visualize how well the model aligns with the true values. This chart helps assess the accuracy 
                 of the linear regression model in predicting the target variable.""")
    else:
        st.error("Please complete the Data Exploration and Analysis step first.")

    


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
