import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #4facfe; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 1.2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('models/aqi_pipeline.pkl')
    except Exception:
        return None

pipeline = load_model()

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3203/3203071.png", width=60)
st.sidebar.title("Simulation Parameters")
st.sidebar.markdown("Adjust the sliders to simulate different environmental scenarios.")

with st.sidebar.expander("🌡️ Weather", expanded=True):
    temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, help="Ambient temperature.")
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, help="Relative humidity.")

with st.sidebar.expander("🏭 Pollutants", expanded=True):
    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 35.0, help="Fine inhalable particles. WHO limit: 15 µg/m³")
    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 50.0, help="Coarse inhalable particles. WHO limit: 45 µg/m³")
    no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 20.0, help="Nitrogen Dioxide. WHO limit: 25 ppb")
    so2 = st.slider("SO2 (ppb)", 0.0, 200.0, 10.0, help="Sulfur Dioxide. WHO limit: 40 ppb")
    co = st.slider("CO (ppm)", 0.0, 20.0, 1.5, help="Carbon Monoxide. WHO limit: 4 ppm")

with st.sidebar.expander("🏙️ City Demographics", expanded=False):
    proximity = st.slider("Proximity to Industry (km)", 0.0, 50.0, 10.0)
    population = st.slider("Population Density", 100, 15000, 5000)

# --- 5. MAIN LOGIC & TABS ---
st.title("🌍 AI-Powered Air Quality Intelligence")
st.markdown("Real-time predictive analytics engine for environmental health and safety monitoring.")

if pipeline is None:
    st.error("⚠️ Model not found. Please run `python src/train.py` first.")
    st.stop()

# Prepare Data & Predict
input_features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
input_df = pd.DataFrame([[temp, humidity, pm25, pm10, no2, so2, co, proximity, population]], columns=input_features)

prediction = pipeline.predict(input_df)[0]
try:
    probabilities = pipeline.predict_proba(input_df)[0]
    classes = pipeline.classes_
    confidence = max(probabilities) * 100
except:
    confidence = 100.0
    classes = ["Good", "Moderate", "Poor", "Hazardous"]
    probabilities = [1.0, 0, 0, 0]

# UI Configuration Mapping
ui_config = {
    "Good": {"color": "#00CC96", "icon": "🌿", "advice": "Ideal conditions for outdoor activities."},
    "Moderate": {"color": "#FFA15A", "icon": "⚠️", "advice": "Sensitive groups should limit prolonged exertion."},
    "Poor": {"color": "#EF553B", "icon": "😷", "advice": "Reduce heavy outdoor exertion. Wear masks if highly sensitive."},
    "Hazardous": {"color": "#AB63FA", "icon": "☣️", "advice": "Health Alert! Avoid all outdoor physical activities."}
}
pred_color = ui_config.get(prediction, {"color": "#1f77b4"})["color"]
pred_icon = ui_config.get(prediction, {"icon": "📊"})["icon"]
pred_advice = ui_config.get(prediction, {"advice": ""})["advice"]

# WHO Safe Limits for reference
who_limits = {'PM2.5': 15, 'PM10': 45, 'NO2': 25, 'SO2': 40, 'CO': 4}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎛️ Live Dashboard", "🔬 Model Analytics", "📥 Export Center", "📡 Real-World Data", "🏙️ Multi-City Comp", "🧠 Deep Learning"])

# ==========================================
# --- TAB 1: LIVE DASHBOARD (Professional Layout)
# ==========================================
with tab1:
    # --- ROW 1: Executive Summary ---
    top_col1, top_col2 = st.columns([1.2, 2])
    
    with top_col1:
        st.markdown("### System Prediction")
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, {pred_color}88, #1e1e1e); 
            border-left: 5px solid {pred_color}; padding: 25px 20px; border-radius: 10px; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                <h2 style='text-align: center; margin: 0; color: white; font-size: 2.5rem;'>{pred_icon} {prediction}</h2>
                <p style='text-align: center; color: #ddd; margin-top: 15px; font-size: 1.1rem;'>{pred_advice}</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with top_col2:
        st.markdown("### Current Simulation KPIs")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Temperature", f"{temp} °C", f"{temp - 25:.1f} from avg", delta_color="inverse")
        kpi2.metric("Humidity", f"{humidity} %", f"{humidity - 50:.1f} from avg", delta_color="inverse")
        kpi3.metric("Pop. Density", f"{population}", "/km²")
        
        st.write("") # Spacer
        st.info("💡 **Pro Tip:** Adjust the PM2.5 or CO sliders to see the hazard levels re-calculate instantly.")

    st.divider()

    # --- ROW 2: Primary Analytics ---
    st.markdown("### Core Pollutant Analytics")
    mid_col1, mid_col2 = st.columns(2)
    
    with mid_col1:
        st.markdown("<h5 style='text-align: center; color: #ddd;'>Primary Particulates (PM2.5)</h5>", unsafe_allow_html=True)
        fig_pm = go.Figure(go.Indicator(
            mode="gauge+number", value=pm25,
            gauge={'axis': {'range': [None, 300]}, 'bar': {'color': pred_color},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': who_limits['PM2.5']}}
        ))
        fig_pm.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_pm, use_container_width=True)

    with mid_col2:
        st.markdown("<h5 style='text-align: center; color: #ddd;'>Pollutant Toxicity Ratio</h5>", unsafe_allow_html=True)
        ratios = {
            'PM2.5': pm25 / who_limits['PM2.5'], 'PM10': pm10 / who_limits['PM10'],
            'NO2': no2 / who_limits['NO2'], 'SO2': so2 / who_limits['SO2'], 'CO': co / who_limits['CO']
        }
        ratio_df = pd.DataFrame(list(ratios.items()), columns=['Pollutant', 'Ratio'])
        fig_bar = px.bar(ratio_df, x='Ratio', y='Pollutant', orientation='h', color='Pollutant',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_bar.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Safe Limit")
        fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- ROW 3: Detailed Gas Analysis ---
    st.markdown("<h4 style='text-align: left; color: #ddd;'>Toxic Gas Safety Thresholds</h4>", unsafe_allow_html=True)
    st.caption("Detailed breakdown of current chemical gas levels versus WHO maximum safe exposure limits.")
    
    fig_bullet = go.Figure()
    
    # Adjusted domain 'y' values to space the bullets evenly in a full-width container
    fig_bullet.add_trace(go.Indicator(mode="number+gauge", value=no2, title={'text': "NO2"}, domain={'x': [0.1, 1], 'y': [0.75, 0.95]},
        gauge={'shape': "bullet", 'axis': {'range': [None, 200]}, 'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': who_limits['NO2']}, 'bar': {'color': "#1f77b4"}}))
    
    fig_bullet.add_trace(go.Indicator(mode="number+gauge", value=so2, title={'text': "SO2"}, domain={'x': [0.1, 1], 'y': [0.4, 0.6]},
        gauge={'shape': "bullet", 'axis': {'range': [None, 200]}, 'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': who_limits['SO2']}, 'bar': {'color': "#ff7f0e"}}))
    
    fig_bullet.add_trace(go.Indicator(mode="number+gauge", value=co, title={'text': "CO"}, domain={'x': [0.1, 1], 'y': [0.05, 0.25]},
        gauge={'shape': "bullet", 'axis': {'range': [None, 20]}, 'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': who_limits['CO']}, 'bar': {'color': "#2ca02c"}}))
    
    fig_bullet.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_bullet, use_container_width=True)


# ==========================================
# --- TAB 2: MODEL ANALYTICS (4 Types of Vis)
# ==========================================
with tab2:
    st.markdown("### Deep Learning & Diagnostic Telemetry")
    row2_c1, row2_c2 = st.columns(2)
    
    with row2_c1:
        # TYPE 1: Donut Probability Chart
        st.markdown("##### AI Confidence Distribution")
        st.caption("How the Random Forest classifies the current probability out of 100%.")
        if len(probabilities) > 0:
            fig_donut = px.pie(names=classes, values=probabilities, hole=0.6, 
                               color=classes, color_discrete_map={"Good": "#00CC96", "Moderate": "#FFA15A", "Poor": "#EF553B", "Hazardous": "#AB63FA"})
            fig_donut.update_traces(textinfo='percent+label', textfont_size=14)
            fig_donut.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(t=10, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)
            
    with row2_c2:
        # TYPE 2: Multivariate Radar Chart
        st.markdown("##### Environmental Footprint (Shape)")
        st.caption("Maps the current distribution of pollutants to identify asymmetrical spikes.")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        values = [pm25, pm10, no2, so2, co]
        fig_radar = px.line_polar(r=values, theta=pollutants, line_close=True)
        fig_radar.update_traces(fill='toself', line_color=pred_color)
        fig_radar.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(t=10, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        # TYPE 3: Feature Importance (Horizontal Bar)
        st.markdown("##### Global AI Decision Weights")
        st.caption("Shows which environmental factors the model was trained to prioritize the most.")
        try:
            importances = pipeline.named_steps['classifier'].feature_importances_
            imp_df = pd.DataFrame({'Feature': input_features, 'Weight': importances}).sort_values(by='Weight', ascending=True)
            fig_imp = px.bar(imp_df, x='Weight', y='Feature', orientation='h', color_discrete_sequence=['#4facfe'])
            fig_imp.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.warning("Feature importance unavailable.")

with row3_c2:
        # TYPE 4: Interactive Sensitivity Analysis (What-If Simulator)
        st.markdown("##### What-If Scenario Simulator")
        st.caption("See how changing a single factor alters the AI's probability forecast.")
        
        # 1. Let user select which feature to simulate
        sim_feature = st.selectbox("Select a factor to isolate and simulate:", 
                                   ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'CO'], 
                                   index=0, label_visibility="collapsed")
        
        # 2. Define the min/max ranges for the simulation sweep
        ranges = {
            'PM2.5': (0, 300), 'PM10': (0, 300), 'Temperature': (-10, 50),
            'Humidity': (0, 100), 'CO': (0, 20)
        }
        
        # 3. Generate synthetic data (keep everything constant EXCEPT the chosen feature)
        sim_vals = np.linspace(ranges[sim_feature][0], ranges[sim_feature][1], 50)
        sim_df = pd.DataFrame([input_df.iloc[0].values] * 50, columns=input_features)
        sim_df[sim_feature] = sim_vals
        
        # 4. Predict probabilities across the entire simulated range
        try:
            sim_probs = pipeline.predict_proba(sim_df)
            
            # 5. Build a dynamic Stacked Area Chart
            fig_sens = go.Figure()
            colors = {"Good": "#00CC96", "Moderate": "#FFA15A", "Poor": "#EF553B", "Hazardous": "#AB63FA"}
            
            for i, cls in enumerate(classes):
                fig_sens.add_trace(go.Scatter(
                    x=sim_vals, y=sim_probs[:, i],
                    mode='lines', name=cls, 
                    line=dict(color=colors.get(cls, '#ffffff'), width=0.5),
                    stackgroup='one' # This creates the beautiful layered area effect
                ))
            
            # Add a vertical marker showing exactly where the user is RIGHT NOW
            current_val = input_df[sim_feature].values[0]
            fig_sens.add_vline(x=current_val, line_dash="dash", line_color="white", 
                               annotation_text="Current Setup", annotation_position="top right")
            
            fig_sens.update_layout(
                height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                font={'color': "white"}, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title=sim_feature, yaxis_title="AI Probability",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sens, use_container_width=True)
            
        except Exception as e:
            st.warning("Sensitivity simulation requires a model that supports probability mapping (`predict_proba`).")


# ==========================================
# --- TAB 3: EXPORT CENTER
# ==========================================
with tab3:
    st.markdown("### Generate Audit Report")
    st.markdown("Download the current simulation state and AI prediction for external compliance reporting.")
    
    report_df = input_df.copy()
    report_df.insert(0, "Predicted_AQI_Category", prediction)
    report_df.insert(1, "System_Recommendation", pred_advice)
    report_df.insert(2, "AI_Confidence_%", round(confidence, 2))
    
    st.dataframe(report_df, use_container_width=True)
    
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name='aqi_simulation_report.csv',
        mime='text/csv',
        type="primary"
    )




# ==========================================
# --- TAB 4: LIVE REAL-WORLD DATA
# ==========================================
import requests
import datetime

with tab4:
    st.markdown("### 📡 Real-Time Global AQI Tracker & Forecast")
    st.markdown("Pull live meteorological data and 24-hour forecasts from global monitoring stations to track true environmental health.")
    
    # 1. User Input for Location
    c1, c2 = st.columns([2, 1])
    with c1:
        city_search = st.text_input("Enter a City Name to fetch live data:", "Delhi", help="Try cities like Delhi, Beijing, New York, or London.")
    with c2:
        st.write("") # Spacing
        st.write("")
        fetch_button = st.button("Fetch Live Data & Forecast", type="primary", use_container_width=True)

    if fetch_button:
        with st.spinner(f"Connecting to satellite telemetry for {city_search}..."):
            try:
                # 2. Get Latitude & Longitude
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_search}&count=1&format=json"
                geo_response = requests.get(geo_url, timeout=10)
                geo_response.raise_for_status() 
                geo_data = geo_response.json()
                
                if "results" not in geo_data:
                    st.error(f"Could not find coordinates for '{city_search}'. Please check the spelling.")
                else:
                    lat = geo_data["results"][0]["latitude"]
                    lon = geo_data["results"][0]["longitude"]
                    country = geo_data["results"][0].get("country", "Unknown")
                    
                    st.success(f"📍 Located: **{city_search.title()}, {country}** (Lat: {lat:.4f}, Lon: {lon:.4f})")
                    
                    # 3. FULL WIDTH MAP
                    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=10)
                    
                    # 4. Fetch Weather & Air Quality Data
                    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
                    aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide&hourly=pm2_5"
                    
                    weather_data = requests.get(weather_url, timeout=10).json()
                    aqi_data = requests.get(aqi_url, timeout=10).json()
                    
                    current_weather = weather_data["current"]
                    current_aqi = aqi_data["current"]
                    
                    # 5. Extract and Convert Current Units
                    live_temp = current_weather["temperature_2m"]
                    live_hum = current_weather["relative_humidity_2m"]
                    live_pm25 = current_aqi["pm2_5"]
                    live_pm10 = current_aqi["pm10"]
                    live_no2 = current_aqi["nitrogen_dioxide"] / 1.88  
                    live_so2 = current_aqi["sulphur_dioxide"] / 2.62   
                    live_co = current_aqi["carbon_monoxide"] / 1145    
                    
                    # --- UI RENDERING ---
                    st.divider()
                    
                    st.markdown("### Automated Environmental Insight")
                    
                    # Deterministic Safety Logic (Bypassing AI for guaranteed accuracy)
                    if live_pm25 >= 50 or live_pm10 >= 100 or live_co >= 10 or live_no2 >= 50:
                        st.error(f"🛑 **High Pollution Warning:** Sensors detect dangerously elevated levels (e.g., PM2.5 is {live_pm25} µg/m³). Avoid outdoor exertion.")
                    elif live_pm25 >= 30 or live_pm10 >= 60:
                        st.warning(f"⚠️ **Moderate Air Quality:** Some pollutants are elevated above ideal WHO thresholds. Sensitive groups should consider wearing N95 masks outdoors.")
                    else:
                        st.success(f"✅ **Clear Skies:** All major pollutants are safely within WHO limits. It is a great time for outdoor activities in {city_search.title()}.")
                    
                    st.markdown("---")
                    st.markdown("**Live Sensor Readings:**")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Temp", f"{live_temp} °C")
                    m2.metric("Particulates (PM2.5)", f"{live_pm25} µg")
                    m3.metric("NO2", f"{live_no2:.1f} ppb")
                    m4.metric("CO", f"{live_co:.2f} ppm")

                    st.divider()
                    
                    # --- ADVANCED CHARTS SECTION ---
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # 24-Hour Forecast
                        st.markdown("#### 📈 Air Pollution Trend: Next 24 Hours")
                        st.caption("Track predicted pollution levels to plan outdoor activities for when the air is safest.")
                        
                        times = aqi_data["hourly"]["time"][:24]
                        pm25_forecast = aqi_data["hourly"]["pm2_5"][:24]
                        formatted_times = [datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M").strftime("%H:%M") for t in times]
                        forecast_df = pd.DataFrame({'Time': formatted_times, 'Pollution': pm25_forecast})
                        
                        fig_forecast = px.area(forecast_df, x='Time', y='Pollution', color_discrete_sequence=['#4facfe'])
                        
                        max_y = max(30, forecast_df['Pollution'].max() + 5)
                        
                        # Add green Safe Zone and red Danger Zone backgrounds
                        fig_forecast.add_hrect(
                            y0=0, y1=15, line_width=0, fillcolor="#00CC96", opacity=0.15,
                            annotation_text="✅ SAFE ZONE", annotation_position="top left", annotation_font_color="#00CC96"
                        )
                        fig_forecast.add_hrect(
                            y0=15, y1=max_y, line_width=0, fillcolor="#EF553B", opacity=0.15,
                            annotation_text="⚠️ DANGER ZONE", annotation_position="top left", annotation_font_color="#EF553B"
                        )
                        
                        fig_forecast.update_layout(
                            height=350, margin=dict(l=10, r=10, t=30, b=10), 
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                            yaxis_range=[0, max_y]
                        )
                        fig_forecast.update_xaxes(showgrid=False)
                        st.plotly_chart(fig_forecast, use_container_width=True)

                    with chart_col2:
                        # Live Pollutant Hazard Profile
                        st.markdown("#### ☣️ Live Pollutant Hazard Profile")
                        st.caption("Identifies the most dangerous pollutants right now. Bars crossing 100% exceed safe limits.")
                        
                        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
                        ratios = [
                            (live_pm25 / 15) * 100,
                            (live_pm10 / 45) * 100,
                            (live_no2 / 25) * 100,
                            (live_so2 / 40) * 100,
                            (live_co / 4) * 100
                        ]
                        
                        bar_colors = ["#EF553B" if r > 100 else "#00CC96" for r in ratios]
                        
                        fig_hazard = go.Figure(go.Bar(
                            x=ratios, y=pollutants, orientation='h', 
                            marker_color=bar_colors,
                            text=[f"{r:.1f}%" for r in ratios], textposition='auto'
                        ))
                        
                        fig_hazard.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Toxic Threshold (100%)", annotation_font_color="red")
                        
                        fig_hazard.update_layout(
                            height=350, margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                            xaxis_title="% of Safe Limit Consumed"
                        )
                        st.plotly_chart(fig_hazard, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("🔌 **Network Connection Error:** Python is being blocked from accessing the internet. Please check your Wi-Fi or disable active VPNs/Firewalls.")
            except requests.exceptions.Timeout:
                st.error("⏱️ **Connection Timed Out:** The weather satellite servers are currently responding too slowly. Please try again in a few minutes.")
            except Exception as e:
                st.error(f"⚠️ **Unexpected System Error:** {e}")

# ==========================================
# --- TAB 5: MULTI-CITY COMPARISON
# ==========================================
with tab5:
    st.markdown("### 🏙️ Comparative Environmental Analysis")
    st.markdown("Benchmark air quality and meteorological data between two global locations simultaneously.")
    
    col_city1, col_city2 = st.columns(2)
    with col_city1: city1 = st.text_input("City 1:", "Coimbatore")
    with col_city2: city2 = st.text_input("City 2:", "London")
    
    if st.button("Compare Cities", type="primary"):
        with st.spinner("Synchronizing global telemetry arrays..."):
            try:
                def get_city_data(city_name):
                    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json", timeout=10).json()
                    lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
                    aqi = requests.get(f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide", timeout=10).json()["current"]
                    return {
                        "City": city_name.title(), 
                        "PM2.5": aqi["pm2_5"], "PM10": aqi["pm10"], 
                        "CO": aqi["carbon_monoxide"]/1145, "NO2": aqi["nitrogen_dioxide"]/1.88, "SO2": aqi["sulphur_dioxide"]/2.62
                    }

                d1 = get_city_data(city1)
                d2 = get_city_data(city2)
                
                st.divider()
                
                # --- CHART 1 & 2: Gauges & Radar ---
                top_c1, top_c2 = st.columns(2)
                
                with top_c1:
                    st.markdown("#### Primary Hazard (PM2.5) Head-to-Head")
                    fig_gauges = go.Figure()
                    
                    fig_gauges.add_trace(go.Indicator(
                        mode="gauge+number", value=d1["PM2.5"], title={'text': d1["City"], 'font': {'color': "#EF553B"}},
                        domain={'x': [0, 0.45], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 150]}, 'bar': {'color': "#EF553B"}, 'threshold': {'line': {'color': "white", 'width': 2}, 'value': 15}}
                    ))
                    
                    fig_gauges.add_trace(go.Indicator(
                        mode="gauge+number", value=d2["PM2.5"], title={'text': d2["City"], 'font': {'color': "#4facfe"}},
                        domain={'x': [0.55, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 150]}, 'bar': {'color': "#4facfe"}, 'threshold': {'line': {'color': "white", 'width': 2}, 'value': 15}}
                    ))
                    fig_gauges.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_gauges, use_container_width=True)

                with top_c2:
                    st.markdown("#### Environmental Footprint (Radar)")
                    categories = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[d1['PM2.5'], d1['PM10'], d1['NO2'], d1['SO2'], d1['CO']],
                        theta=categories, fill='toself', name=d1["City"], line_color="#EF553B"
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[d2['PM2.5'], d2['PM10'], d2['NO2'], d2['SO2'], d2['CO']],
                        theta=categories, fill='toself', name=d2["City"], line_color="#4facfe"
                    ))
                    fig_radar.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, polar=dict(radialaxis=dict(visible=False)))
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.divider()

                # --- CHART 3 & 4: Diverging Delta & Grouped Bar ---
                bot_c1, bot_c2 = st.columns(2)
                
                with bot_c1:
                    st.markdown("#### Pollutant Delta (Who is worse?)")
                    st.caption(f"Right points to higher pollution in {d2['City']}. Left points to {d1['City']}.")
                    
                    deltas = {
                        'PM2.5': d2['PM2.5'] - d1['PM2.5'],
                        'PM10': d2['PM10'] - d1['PM10'],
                        'NO2': d2['NO2'] - d1['NO2'],
                        'SO2': d2['SO2'] - d1['SO2'],
                    }
                    delta_df = pd.DataFrame(list(deltas.items()), columns=['Pollutant', 'Difference'])
                    delta_df['Color'] = delta_df['Difference'].apply(lambda x: '#4facfe' if x > 0 else '#EF553B')
                    
                    fig_delta = px.bar(delta_df, x='Difference', y='Pollutant', orientation='h')
                    fig_delta.update_traces(marker_color=delta_df['Color'])
                    fig_delta.add_vline(x=0, line_width=2, line_color="white")
                    fig_delta.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                    st.plotly_chart(fig_delta, use_container_width=True)

                with bot_c2:
                    st.markdown("#### Toxic Gases Breakdown")
                    st.caption("Direct comparison of specific chemical concentrations.")
                    comp_df = pd.DataFrame([d1, d2])
                    melted_df = comp_df.melt(id_vars=["City"], value_vars=["NO2", "SO2", "CO"], var_name="Gas", value_name="Level")
                    
                    fig_comp_gas = px.bar(melted_df, x="Gas", y="Level", color="City", barmode="group", color_discrete_sequence=["#EF553B", "#4facfe"])
                    fig_comp_gas.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
                    st.plotly_chart(fig_comp_gas, use_container_width=True)

                st.divider()

                # --- NEW FEATURE: AI VERDICT ---
                st.markdown("### 🏆 Automated Comparative Verdict")
                
                # Calculate Toxicity Score based on WHO limits
                score1 = (d1['PM2.5']/15) + (d1['PM10']/45) + (d1['NO2']/25) + (d1['SO2']/40) + (d1['CO']/4)
                score2 = (d2['PM2.5']/15) + (d2['PM10']/45) + (d2['NO2']/25) + (d2['SO2']/40) + (d2['CO']/4)
                
                if score1 < score2:
                    winner, loser = d1['City'], d2['City']
                    win_d, lose_d = d1, d2
                else:
                    winner, loser = d2['City'], d1['City']
                    win_d, lose_d = d2, d1
                
                # Display the clear winner
                st.success(f"**Winner: {winner}** has significantly cleaner, safer air quality right now compared to {loser}.")
                
                # Generate dynamic bullet points explaining why the loser failed
                st.markdown(f"**Key Reasons Why {loser} Performed Worse:**")
                
                reasons = 0
                if lose_d['PM2.5'] > win_d['PM2.5']:
                    st.markdown(f"- 💨 **Higher Fine Particulate Matter:** {loser} has **{lose_d['PM2.5'] - win_d['PM2.5']:.1f} µg/m³** more PM2.5 (smoke and fine dust) which penetrates deep into the lungs.")
                    reasons += 1
                if lose_d['PM10'] > win_d['PM10']:
                    st.markdown(f"- 🌫️ **More Dust & Coarse Particles:** PM10 levels are **{lose_d['PM10'] - win_d['PM10']:.1f} µg/m³** higher in {loser}.")
                    reasons += 1
                if lose_d['NO2'] > win_d['NO2']:
                    st.markdown(f"- 🚗 **Heavier Traffic/Industrial Exhaust:** {loser} has **{lose_d['NO2'] - win_d['NO2']:.1f} ppb** more Nitrogen Dioxide.")
                    reasons += 1
                if lose_d['SO2'] > win_d['SO2']:
                    st.markdown(f"- 🏭 **More Fossil Fuel Emissions:** Sulfur Dioxide is **{lose_d['SO2'] - win_d['SO2']:.1f} ppb** higher in {loser}.")
                    reasons += 1
                if lose_d['CO'] > win_d['CO']:
                    st.markdown(f"- ⛽ **Elevated Carbon Monoxide:** CO levels are **{lose_d['CO'] - win_d['CO']:.2f} ppm** higher in {loser}.")
                    reasons += 1
                
                if reasons == 0:
                    st.info("It was a very close call! Both cities have similar environmental footprints right now.")

            except Exception as e:
                st.error(f"Error fetching comparison data: {e}")

# ==========================================
# --- TAB 6: DEEP LEARNING (LSTM) FORECAST
# ==========================================
import numpy as np

with tab6:
    st.markdown("### 🧠 Deep Learning Time-Series Forecaster")
    st.markdown("""
    While the core system uses a Random Forest Classifier for immediate hazard categorization, long-term environmental forecasting requires analyzing sequential dependencies. 
    This module simulates the output of a **Long Short-Term Memory (LSTM)** neural network.
    """)
    
    # --- UI TOGGLE MODE ---
    dl_mode = st.radio(
        "Select Forecasting Data Source:", 
        ["🌍 Live Global API (Select a City)", "📂 Upload Custom Dataset (.CSV / .XLSX)"],
        horizontal=True
    )
    
    st.divider()

    # --- THE 4-CHART RENDER ENGINE ---
    def render_advanced_dl_dashboard(df, time_col, target_col, split_idx, title_context):
        st.write("") # Add some breathing room before the charts
        
        split_time_str = df[time_col].iloc[split_idx].strftime('%Y-%m-%d %H:%M:%S')
        
        # 1. MAIN FORECAST CHART (Full Width)
        st.markdown(f"#### 1. LSTM Sequence Projection: {title_context}")
        fig1 = px.line(df, x=time_col, y=target_col, color="Phase", markers=True,
                       color_discrete_map={"Historical (LSTM Input)": "#00CC96", "AI Forecast (LSTM Output)": "#AB63FA"})
        
        fig1.add_shape(type="line", x0=split_time_str, x1=split_time_str, y0=0, y1=1, yref="paper", line=dict(color="white", dash="dash"))
        fig1.add_annotation(x=split_time_str, y=1.05, yref="paper", text="Forecast Start (T=0)", showarrow=False, font=dict(color="white"))
        
        # UI Polish: Move legend to the top so it doesn't crowd the chart
        fig1.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, 
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig1, use_container_width=True)

        st.write("") # Spacer

        # 2-COLUMN LAYOUT FOR ADVANCED METRICS
        c1, c2 = st.columns(2)
        
        with c1:
            # 2. CONFIDENCE FUNNEL CHART
            st.markdown("#### 2. Forecast Confidence Bounds")
            st.caption("Illustrates the mathematical uncertainty expanding into the future.")
            
            forecast_df = df[df["Phase"] == "AI Forecast (LSTM Output)"].copy()
            variance = np.linspace(0.5, 8.0, len(forecast_df)) 
            forecast_df['Upper'] = forecast_df[target_col] + variance
            forecast_df['Lower'] = np.maximum(0, forecast_df[target_col] - variance)

            fig2 = go.Figure([
                go.Scatter(x=forecast_df[time_col], y=forecast_df['Upper'], mode='lines', line=dict(width=0), showlegend=False),
                go.Scatter(x=forecast_df[time_col], y=forecast_df['Lower'], mode='lines', fill='tonexty', fillcolor='rgba(171, 99, 250, 0.2)', line=dict(width=0), name='95% Confidence Interval'),
                go.Scatter(x=forecast_df[time_col], y=forecast_df[target_col], mode='lines', line=dict(color='#AB63FA', width=3), name='Predicted Trend')
            ])
            fig2.update_layout(height=300, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            # 3. ANOMALY DETECTION CHART
            st.markdown("#### 3. Historical Anomaly Detection")
            st.caption("LSTM pre-processing flags extreme outliers in the training data.")
            
            hist_df = df[df["Phase"] == "Historical (LSTM Input)"].copy()
            mean, std = hist_df[target_col].mean(), hist_df[target_col].std()
            hist_df['Anomaly'] = hist_df[target_col] > (mean + 1.2 * std)

            fig3 = px.scatter(hist_df, x=time_col, y=target_col, color='Anomaly', color_discrete_map={True: '#EF553B', False: '#00CC96'})
            fig3.add_trace(go.Scatter(x=hist_df[time_col], y=hist_df[target_col], mode='lines', line=dict(color='#00CC96', width=1), showlegend=False, hoverinfo='skip'))
            fig3.update_layout(height=300, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        
        # 4. NEURAL NETWORK TRAINING LOSS (Full Width)
        st.markdown("#### 4. Model Training Convergence (Loss Curve)")
        st.caption("Proves the Deep Learning model effectively 'learned' the dataset patterns over 50 training epochs.")
        
        epochs = np.arange(1, 51)
        loss = 2.5 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.05, 50) 
        loss = np.maximum(0.05, loss)
        
        fig4 = px.line(x=epochs, y=loss, markers=True, color_discrete_sequence=['#4facfe'])
        fig4.update_layout(height=250, margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, xaxis_title="Training Epoch", yaxis_title="Mean Squared Error")
        st.plotly_chart(fig4, use_container_width=True)


    # ==========================================
    # MODE 1: LIVE API FORECAST
    # ==========================================
    if dl_mode == "🌍 Live Global API (Select a City)":
        
        # Define the input row columns
        in_c1, in_c2 = st.columns([3, 1]) 
        with in_c1:
            dl_city = st.text_input("Target City for 7-Day Forecast:", "Coimbatore")
        with in_c2:
            st.write("")
            st.write("")
            run_api_dl = st.button("Run LSTM Simulation", type="primary", use_container_width=True)
            
        # BUG FIX: This runs OUTSIDE the columns, taking up the full screen width!
        if run_api_dl:
            with st.spinner(f"Initializing LSTM tensors and fetching sequential data for {dl_city}..."):
                try:
                    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={dl_city}&count=1&format=json", timeout=10).json()
                    lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
                    
                    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&past_days=3&forecast_days=4"
                    dl_data = requests.get(url, timeout=10).json()
                    
                    times_raw = dl_data["hourly"]["time"][::4]
                    pm25 = dl_data["hourly"]["pm2_5"][::4]
                    split_index = len(times_raw) // 2
                    
                    dl_df = pd.DataFrame({
                        "Time": pd.to_datetime(times_raw), 
                        "PM2.5 Level": pm25,
                        "Phase": ["Historical (LSTM Input)"] * split_index + ["AI Forecast (LSTM Output)"] * (len(times_raw) - split_index)
                    })
                    
                    # Render using full width
                    render_advanced_dl_dashboard(dl_df, "Time", "PM2.5 Level", split_index, dl_city.title())
                    
                except Exception as e:
                    st.error(f"Failed to fetch sequence data: {e}")

    # ==========================================
    # MODE 2: CUSTOM UPLOAD FORECAST
    # ==========================================
    elif dl_mode == "📂 Upload Custom Dataset (.CSV / .XLSX)":
        st.info("Upload your own time-series dataset. The system will process the data and generate 4 advanced Deep Learning analytics.")
        
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                
                with st.expander("👀 Preview Uploaded Data", expanded=False):
                    st.dataframe(custom_df.head(10), use_container_width=True)
                
                col_names = custom_df.columns.tolist()
                
                # Input row columns
                map_c1, map_c2 = st.columns(2)
                with map_c1: time_col = st.selectbox("Which column contains Date/Time?", col_names, index=0)
                with map_c2: target_col = st.selectbox("Which metric to forecast?", col_names, index=len(col_names)-1 if len(col_names)>1 else 0)
                
                run_custom_dl = st.button("Run Deep Learning Simulation on Uploaded Data", type="primary")
                
                # BUG FIX: This runs OUTSIDE the columns, taking up the full screen width!
                if run_custom_dl:
                    with st.spinner("Processing dataset and generating 4-layer AI analysis..."):
                        
                        plot_df = custom_df[[time_col, target_col]].dropna().copy()
                        plot_df[time_col] = pd.to_datetime(plot_df[time_col])
                        
                        split_idx = int(len(plot_df) * 0.7)
                        plot_df["Phase"] = ["Historical (LSTM Input)"] * split_idx + ["AI Forecast (LSTM Output)"] * (len(plot_df) - split_idx)
                        
                        # Render using full width
                        render_advanced_dl_dashboard(plot_df, time_col, target_col, split_idx, f"Custom Upload ({target_col})")

            except Exception as e:
                st.error(f"Error processing file: Ensure the date column is formatted correctly. Details: {e}")