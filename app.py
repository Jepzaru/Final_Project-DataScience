
import streamlit as st

st.set_page_config(
    page_title="Final Project",  
    page_icon="sjlogo.png",  
    layout="wide"  
)


def sidebar():

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
    </style>
    """
    st.markdown(navlink_style, unsafe_allow_html=True)
    
    st.sidebar.image('sjlogo.png', width=400) 
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="nav-link">⚪ Home</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="nav-link">⚪ About</a>', unsafe_allow_html=True)
    st.sidebar.markdown('<a href="#" class="nav-link">⚪ Contact</a>', unsafe_allow_html=True)


sidebar()

st.title('SuperStore Sales')
st.markdown('<hr>', unsafe_allow_html=True)
st.image('salestore.jpg', use_container_width=True)

st.title('Content')
st.markdown("""
<style>
p {
    font-size: 26px; 
}
</style>
Time series analysis deals with time series based data to extract patterns for predictions 
and other characteristics of the data. It uses a model for forecasting future values in a small time 
frame based on previous observations. It is widely used for non-stationary data, such as economic data, 
weather data, stock prices, and retail sales forecasting.
""", unsafe_allow_html=True)


