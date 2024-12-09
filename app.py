import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Final Project",  
    page_icon="sjlogo.png",  
    layout="wide"  
)

def sidebar():
    current_page = st.session_state.get('page', 'overview')  # Default to 'overview' if not set
    navlink_style = """
    <style>
    .nav-link {
        display: block;
        padding: 10px;
        color: white;
        font-size: 24px;
        text-decoration: none;
        border-radius: 10px;
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
    </style>
    """
    st.markdown(navlink_style, unsafe_allow_html=True)
    
    st.sidebar.image('sjlogo.png', width=400) 
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    overview_class = 'active' if current_page == 'overview' else ''
    st.sidebar.markdown(f'<a href="#" class="nav-link {overview_class}">⚪ Overview</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="nav-link">⚪ Analytics</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="nav-link">⚪ Data Visualization</a>', unsafe_allow_html=True)

sidebar()

st.session_state['page'] = 'overview'

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
