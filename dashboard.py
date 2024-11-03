import streamlit as st
import pandas as pd
import plotly.express as px
import openai
from streamlit_plotly_events import plotly_events
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic
from streamlit_folium import st_folium

# Set Streamlit layout to wide
st.set_page_config(layout="wide")
# Text Preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Request OpenAI API Key input from the user
language = st.sidebar.selectbox("Select your language", ("English","Indonesia"))
if language == "English":
    api_key = st.sidebar.text_input("Enter your GROK API Key", type="password")
if language == "Indonesia":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Set OpenAI API Key if provided
if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your API key to continue.")

# Load processed data
@st.cache_data
def load_data():
    data_path = './data/processed_news_data.csv'
    data = pd.read_csv(data_path)
    data['Pub_Date'] = pd.to_datetime(data['Pub_Date'], errors='coerce')  # Keep full datetime for deduplication

    # Filter only rows where Conflict Escalation is TRUE
    data = data[data['Conflict Escalation'] == True]

    # Remove duplicates based on 'News_Title' while keeping the latest news by date
    data = data.sort_values('Pub_Date').drop_duplicates(subset='News_Title', keep='last')
    
    # Convert 'Pub_Date' to date only for other processing
    data['Pub_Date'] = data['Pub_Date'].dt.date
    
    return data
#Perform Topic Modelling
def preprocess_text(text):
    # Remove special characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    # Tokenization and remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#Read the Data
data = load_data()
data['processed_text'] = data['News_Title'].apply(preprocess_text)

# Display preprocessed data
st.write("Sample of Preprocessed Data", data[['processed_text']].head())

# Generate Word Cloud
st.write("### Word Cloud")
all_text = ' '.join(data['processed_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
st.pyplot(plt)

# Vectorize the text data for LDA
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
text_vectorized = vectorizer.fit_transform(data['processed_text'])

# Fit LDA model
n_topics = 5
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(text_vectorized)

# Display top words per topic
# Generate LDA visualization with pyLDAvis
st.write("### LDA Topic Model Visualization")

with st.spinner("Generating visualization..."):
    lda_vis = pyLDAvis.lda_model.prepare(lda_model, text_vectorized, vectorizer, mds='tsne')
    pyLDAvis.save_html(lda_vis, './data/lda_vis.html')
    st.components.v1.html(open('./data/lda_vis.html', 'r').read(), width=1300, height=800)

#Get Safe Haven Place

# Initialize OpenCage with your API key
api_key = "9b19285ea138459cb863a910c24d7cfc"  # Replace with your OpenCage API key
geocoder = OpenCageGeocode(api_key)

# List of KJRI and KBRI locations around Australia, USA, Russia, Ukraine, China, and the Middle East
embassy_locations = [
    # Australia
    {"name": "KBRI Canberra", "latitude": -35.282, "longitude": 149.1286, "city": "Canberra, Australia"},
    {"name": "KJRI Sydney", "latitude": -33.8675, "longitude": 151.207, "city": "Sydney, Australia"},
    {"name": "KJRI Melbourne", "latitude": -37.8136, "longitude": 144.9631, "city": "Melbourne, Australia"},
    {"name": "KJRI Perth", "latitude": -31.9505, "longitude": 115.8605, "city": "Perth, Australia"},

    # USA
    {"name": "KBRI Washington, D.C.", "latitude": 38.9101, "longitude": -77.0517, "city": "Washington, D.C., USA"},
    {"name": "KJRI New York", "latitude": 40.749688, "longitude": -73.968549, "city": "New York, USA"},
    {"name": "KJRI Los Angeles", "latitude": 34.0501, "longitude": -118.2551, "city": "Los Angeles, USA"},
    {"name": "KJRI Houston", "latitude": 29.7604, "longitude": -95.3698, "city": "Houston, USA"},

    # Russia
    {"name": "KBRI Moscow", "latitude": 55.7558, "longitude": 37.6173, "city": "Moscow, Russia"},

    # Ukraine
    {"name": "KBRI Kyiv", "latitude": 50.4501, "longitude": 30.5234, "city": "Kyiv, Ukraine"},

    # China
    {"name": "KBRI Beijing", "latitude": 39.9042, "longitude": 116.4074, "city": "Beijing, China"},
    {"name": "KJRI Guangzhou", "latitude": 23.1291, "longitude": 113.2644, "city": "Guangzhou, China"},
    {"name": "KJRI Shanghai", "latitude": 31.2304, "longitude": 121.4737, "city": "Shanghai, China"},
    {"name": "KJRI Hong Kong", "latitude": 22.3193, "longitude": 114.1694, "city": "Hong Kong, China"},

    # Middle East
    {"name": "KBRI Riyadh", "latitude": 24.7136, "longitude": 46.6753, "city": "Riyadh, Saudi Arabia"},
    {"name": "KJRI Jeddah", "latitude": 21.4858, "longitude": 39.1925, "city": "Jeddah, Saudi Arabia"},
    {"name": "KBRI Abu Dhabi", "latitude": 24.4539, "longitude": 54.3773, "city": "Abu Dhabi, UAE"},
    {"name": "KJRI Dubai", "latitude": 25.276987, "longitude": 55.296249, "city": "Dubai, UAE"},
]

st.title("Find Nearest Indonesian Embassy or Consulate")

# Get user input for city
user_city = st.text_input("Enter your city location:", "")

if user_city:
    try:
        # Get coordinates for user's city
        result = geocoder.geocode(user_city)
        if result:
            user_coords = (result[0]['geometry']['lat'], result[0]['geometry']['lng'])

            # Find the closest embassy or consulate based on geodesic distance
            closest_location = min(
                embassy_locations,
                key=lambda loc: geodesic(user_coords, (loc["latitude"], loc["longitude"])).kilometers
            )

            closest_coords = (closest_location["latitude"], closest_location["longitude"])
            distance = geodesic(user_coords, closest_coords).kilometers
            st.write(f"Nearest Indonesian Embassy or Consulate: {closest_location['name']} in {closest_location['city']}")
            st.write(f"Distance: {distance:.2f} km")

            # Create map centered around user location and nearest embassy
            map_center = [(user_coords[0] + closest_coords[0]) / 2, (user_coords[1] + closest_coords[1]) / 2]
            folium_map = folium.Map(location=map_center, zoom_start=6)

            # Add user location marker
            folium.Marker(
                user_coords, tooltip="Your Location", popup=f"{user_city}", icon=folium.Icon(color="blue")
            ).add_to(folium_map)

            # Add nearest embassy location marker
            folium.Marker(
                closest_coords, tooltip=closest_location["name"], popup=closest_location["city"], icon=folium.Icon(color="green")
            ).add_to(folium_map)

            # Draw a line between user location and nearest embassy
            folium.PolyLine([user_coords, closest_coords], color="blue", weight=2.5, opacity=1).add_to(folium_map)

            # Display map
            st_folium(folium_map, width=700, height=500)
        else:
            st.error("Could not geocode the specified city. Please enter a valid city name.")

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Summarize data by country within date range
def prepare_choropleth_data(data, date_range):
    filtered_data = data[(data['Pub_Date'] >= date_range[0]) & (data['Pub_Date'] <= date_range[1])]
    if 'Countries Mentioned' in filtered_data.columns:
        country_counts = filtered_data['Countries Mentioned'].fillna('Unknown').value_counts().reset_index()
        country_counts.columns = ['Country', 'News_Count']
        return country_counts
    else:
        raise ValueError("'Countries Mentioned' column not found in the dataset")

# Sidebar menu buttons
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = "Dashboard"  # Default to "Dashboard"

with st.sidebar:
    if st.button("Paper", use_container_width=True):  # Full-width button
        st.session_state.menu_selection = "Paper"
    if st.button("Dashboard", use_container_width=True):  # Full-width button
        st.session_state.menu_selection = "Dashboard"

# Display content based on the selected menu
if st.session_state.menu_selection == "Paper":
    st.title("Paper and Documentation")
    # Placeholder for paper/documentation content
    st.write("Content for paper and documentation goes here.")

elif st.session_state.menu_selection == "Dashboard":
    st.title("Conflict Monitoring Dashboard")

    try:
        # Load data
        data = load_data()
        st.write(data.head())
        st.write(data.shape)
        # Date range slider with only date format (no time)
        min_date = data['Pub_Date'].min()
        max_date = data['Pub_Date'].max()
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(max_date, max_date),  # Set default to the last day only
            format="YYYY-MM-DD"
        )

        # Prepare data for the map
        choropleth_data = prepare_choropleth_data(data, date_range)
        
        # Dynamically set the color scale maximum based on the date range's highest news count
        max_news_count = choropleth_data["News_Count"].max()

        # Display choropleth map
        st.header("Global Conflict News Counts by Country")
        fig = px.choropleth(
            choropleth_data,
            locations="Country",
            locationmode="country names",
            color="News_Count",
            color_continuous_scale="Reds",
            title="Number of Conflict News Mentions by Country",
            labels={'News_Count': 'News Articles Count'}
        )

        # Set the color bar range
        fig.update_layout(coloraxis_colorbar=dict(ticksuffix=""), coloraxis_cmax=max_news_count)
        fig.update_geos(showcoastlines=True, coastlinecolor="Gray")

        # Display the map and handle clicks
        selected_points = plotly_events(fig, click_event=True, hover_event=False)

        # Display the top 3 countries with the most conflict news below the map and above the news table
        top_countries = choropleth_data.nlargest(3, 'News_Count')
        top_countries_display = ", ".join([f"{row['Country']}({row['News_Count']})" for _, row in top_countries.iterrows()])
        st.write(f"Countries with most news about conflict: {top_countries_display}")

        # Handle map click to select a country
        if selected_points and "pointIndex" in selected_points[0]:
            # Retrieve the clicked country based on point index
            point_index = selected_points[0]["pointIndex"]
            clicked_country = choropleth_data.iloc[point_index]["Country"]
            st.session_state["selected_country"] = clicked_country

        if "selected_country" in st.session_state:
            clicked_country = st.session_state["selected_country"]
            
            st.subheader(f"News Articles for {clicked_country} between {date_range[0]} and {date_range[1]}")

            # Filter news for the clicked country and date range
            mask = (
                data['Countries Mentioned'].str.contains(clicked_country, case=False, na=False) &
                (data['Pub_Date'] >= date_range[0]) & 
                (data['Pub_Date'] <= date_range[1])
            )
            country_news = data[mask]

            # Pagination setup
            items_per_page = 10
            total_pages = (len(country_news) + items_per_page - 1) // items_per_page
            
            # Display pagination control above the table
            page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, value=1)
            start = (page - 1) * items_per_page
            end = start + items_per_page
            paginated_news = country_news.iloc[start:end]
            
            # Display paginated news table with clickable URL
            if len(paginated_news) > 0:
                # Display news articles in a table format
                st.write(
                    paginated_news[['Pub_Date', 'News_Title', 'URL']].to_html(
                        index=False, 
                        escape=False, 
                        formatters={'URL': lambda x: f'<a href="{x}" target="_blank">Link</a>'}
                    ), 
                    unsafe_allow_html=True
                )
            else:
                st.write("No news articles found for the selected criteria.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data format and try again.")

# Function to call OpenAI API for analysis with the new API structure

# Define the OpenAI conflict escalation analysis function
def get_conflict_analysis(country):
    # Create a structured prompt for OpenAI
    prompt = (
        f"As part of an analysis for the Ministry of Foreign Affairs, provide an assessment on the probability of conflict escalation in {country}. "
        f"Refer to any typical indicators or patterns related to economic or political tensions in similar situations, as well as any regional dependencies. "
        f"Provide a structured report as follows: "
        f"- **Situation Overview**: Describe any relevant political or economic background for {country}. "
        f"- **Risk of Fiscal Impact**: Assess the potential economic impact of escalation on government fiscal stability. "
        f"- **Evacuation Considerations**: Estimate the number of Indonesian nationals who might need evacuation and any specific logistical challenges. "
        f"- **Conflict Prevention Recommendations**: Provide recommendations to reduce the risk of escalation or mitigate impacts."
    )

    # Call OpenAI API to generate response using the ChatCompletion endpoint
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use gpt-4 or gpt-3.5-turbo based on your access
        messages=[
            {"role": "system", "content": "You are an expert analyst for conflict assessment."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()

# Dropdown and analysis display below the news table
st.subheader("Choose a country for further conflict escalation analysis")
chosen_country = st.selectbox("Select a country for analysis", choropleth_data["Country"].unique())

if chosen_country:
    st.subheader(f"Conflict Escalation Analysis for {chosen_country}")
    analysis = get_conflict_analysis(chosen_country)
    if analysis:
        st.write(analysis)

import streamlit as st
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium

# Set up geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

# List of KJRI and KBRI locations around Australia, USA, Russia, Ukraine, China, and the Middle East
embassy_locations = [
    # Australia
    {"name": "KBRI Canberra", "latitude": -35.282, "longitude": 149.1286, "city": "Canberra, Australia"},
    {"name": "KJRI Sydney", "latitude": -33.8675, "longitude": 151.207, "city": "Sydney, Australia"},
    {"name": "KJRI Melbourne", "latitude": -37.8136, "longitude": 144.9631, "city": "Melbourne, Australia"},
    {"name": "KJRI Perth", "latitude": -31.9505, "longitude": 115.8605, "city": "Perth, Australia"},

    # USA
    {"name": "KBRI Washington, D.C.", "latitude": 38.9101, "longitude": -77.0517, "city": "Washington, D.C., USA"},
    {"name": "KJRI New York", "latitude": 40.749688, "longitude": -73.968549, "city": "New York, USA"},
    {"name": "KJRI Los Angeles", "latitude": 34.0501, "longitude": -118.2551, "city": "Los Angeles, USA"},
    {"name": "KJRI Houston", "latitude": 29.7604, "longitude": -95.3698, "city": "Houston, USA"},

    # Russia
    {"name": "KBRI Moscow", "latitude": 55.7558, "longitude": 37.6173, "city": "Moscow, Russia"},

    # Ukraine
    {"name": "KBRI Kyiv", "latitude": 50.4501, "longitude": 30.5234, "city": "Kyiv, Ukraine"},

    # China
    {"name": "KBRI Beijing", "latitude": 39.9042, "longitude": 116.4074, "city": "Beijing, China"},
    {"name": "KJRI Guangzhou", "latitude": 23.1291, "longitude": 113.2644, "city": "Guangzhou, China"},
    {"name": "KJRI Shanghai", "latitude": 31.2304, "longitude": 121.4737, "city": "Shanghai, China"},
    {"name": "KJRI Hong Kong", "latitude": 22.3193, "longitude": 114.1694, "city": "Hong Kong, China"},

    # Middle East
    {"name": "KBRI Riyadh", "latitude": 24.7136, "longitude": 46.6753, "city": "Riyadh, Saudi Arabia"},
    {"name": "KJRI Jeddah", "latitude": 21.4858, "longitude": 39.1925, "city": "Jeddah, Saudi Arabia"},
    {"name": "KBRI Abu Dhabi", "latitude": 24.4539, "longitude": 54.3773, "city": "Abu Dhabi, UAE"},
    {"name": "KJRI Dubai", "latitude": 25.276987, "longitude": 55.296249, "city": "Dubai, UAE"},
]

st.title("Find Nearest Indonesian Embassy or Consulate")

# Get user input for city
user_city = st.text_input("Enter your city location:", "")

if user_city:
    try:
        # Get coordinates for user's city
        user_location = geolocator.geocode(user_city)
        user_coords = (user_location.latitude, user_location.longitude)
        
        # Find the closest embassy or consulate based on geodesic distance
        closest_location = min(
            embassy_locations,
            key=lambda loc: geodesic(user_coords, (loc["latitude"], loc["longitude"])).kilometers
        )
        
        closest_coords = (closest_location["latitude"], closest_location["longitude"])
        distance = geodesic(user_coords, closest_coords).kilometers
        st.write(f"Nearest Indonesian Embassy or Consulate: {closest_location['name']} in {closest_location['city']}")
        st.write(f"Distance: {distance:.2f} km")

        # Create map centered around user location and nearest embassy
        map_center = [(user_coords[0] + closest_coords[0]) / 2, (user_coords[1] + closest_coords[1]) / 2]
        folium_map = folium.Map(location=map_center, zoom_start=6)

        # Add user location marker
        folium.Marker(
            user_coords, tooltip="Your Location", popup=f"{user_city}", icon=folium.Icon(color="blue")
        ).add_to(folium_map)

        # Add nearest embassy location marker
        folium.Marker(
            closest_coords, tooltip=closest_location["name"], popup=closest_location["city"], icon=folium.Icon(color="green")
        ).add_to(folium_map)

        # Draw a line between user location and nearest embassy
        folium.PolyLine([user_coords, closest_coords], color="blue", weight=2.5, opacity=1).add_to(folium_map)

        # Display map
        st_folium(folium_map, width=700, height=500)

    except AttributeError:
        st.error("Could not find the specified city. Please enter a valid city name.")
else:
    st.info("Please enter a city name to see directions to the nearest Indonesian Embassy or Consulate.")

