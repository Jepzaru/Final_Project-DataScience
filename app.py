import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Final Project",  
    page_icon="sjlogo.png",  
    layout="wide"  
)

def sidebar():
    # Get the current page from the session state or default to 'overview'
    current_page = st.session_state.get('page', 'overview')
    
    # Add CSS for styling the sidebar navigation links
    navlink_style = """
    <style>
    .nav-link {
        display: block;
        padding: 10px;
        color: white;
        font-size: 24px;
        text-decoration: none;
        border-radius: 10px;
        text-align: center;
        width: 100%; /* Full width to make all buttons equal */
        box-sizing: border-box; /* Ensure padding is included in width */
    }
    .nav-link:hover {
        background-color: black;
        text-decoration: none;
        font-weight: 700;
    }
    .active {
        background-color: black;
        font-weight: 700;
    }
    .sidebar-button-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """
    st.markdown(navlink_style, unsafe_allow_html=True)

    # Sidebar content
    st.sidebar.image('sjlogo.png', width=400)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

    # Define navigation links and their corresponding page keys
    nav_links = {
        "overview": "⚪ Overview &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
        "analytics": "⚪ Data Exploration&nbsp;&nbsp;&nbsp;&nbsp;",
        "data_viz": "⚪ Data Visualization&nbsp;"
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
    st.session_state['page'] = 'overview'

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

    # Display dataset structure
    st.subheader('Dataset Structure')
    df = pd.read_csv('train.csv')
    st.dataframe(df.head())

elif current_page == 'analytics':
    st.title('Data Exploration')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    ### Data Exploration
    - Clean the data.
    - Handle missing values.
    - Perform basic descriptive statistics.
    """)

    # Data Exploration
    st.subheader('Data Cleaning and Exploration')
    # Display the initial state of the dataframe
    st.write("### Initial Data")
    st.dataframe(df)

    # Handle missing values
    st.write("### Handling Missing Values")
    df_clean = df.dropna()  # Drop rows with missing values
    # or use fillna() for a different approach
    # df_clean = df.fillna(method='ffill')  # Forward fill missing values

    # Basic Descriptive Statistics
    st.write("### Descriptive Statistics")
    st.write(df_clean.describe())

    # Example: Showing distribution of 'Sales' column
    st.write("### Sales Distribution")
    fig, ax = plt.subplots()
    df_clean['Sales'].hist(bins=30, ax=ax)
    ax.set_title('Sales Distribution')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif current_page == 'data_viz':
    st.title('Data Visualization')
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    ### Data Visualization Page
    - Display charts and graphs.
    - Explore data interactively.
    - Understand key metrics visually.
    """)
