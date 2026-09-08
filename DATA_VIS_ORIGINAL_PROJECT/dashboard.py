"""
Interactive Climate Data Dashboard
Streamlit Application for Weather Data Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ...existing code...
# Ensure we look in the assets/ folder for your CSS

st.set_page_config(
    page_title="Climate Data Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Darker metric cards so light text is legible */
    .stMetric {
        background-color: #0f1720 !important;
        color: #e6eef6 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }
    /* Ensure metric title and value inherit readable colors */
    .stMetric div, .stMetric span, .stMetric p {
        color: #e6eef6 !important;
    }
    /* Story mode styling */
    .story-chapter {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .climate-fact {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌍 Global Weather & Climate Analysis Dashboard")
st.markdown("**Interactive visualization of NOAA weather data (2006-2016)**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        df_daily = pd.read_csv( 'Data_Aggregation_csv\data_cleaned_daily.csv')
        df_monthly = pd.read_csv('Data_Aggregation_csv\data_cleaned_monthly.csv')
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
        return df_daily, df_monthly
    except FileNotFoundError:
        st.error("⚠️ Data files not found. Please run main_analysis.py first to generate cleaned data.")
        st.stop()

df_daily, df_monthly = load_data()

# Sidebar filters
st.sidebar.header("📊 Dashboard Controls")

# Date range filter
min_date = df_daily['Date'].min().date()
max_date = df_daily['Date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_daily_filtered = df_daily[(df_daily['Date'].dt.date >= start_date) & 
                                   (df_daily['Date'].dt.date <= end_date)]
    df_monthly_filtered = df_monthly[(df_monthly['Date'].dt.date >= start_date) & 
                                       (df_monthly['Date'].dt.date <= end_date)]
else:
    df_daily_filtered = df_daily
    df_monthly_filtered = df_monthly

# Season filter
seasons = ['All'] + sorted(df_daily['Season'].unique().tolist())
selected_season = st.sidebar.selectbox("Select Season", seasons)

if selected_season != 'All':
    df_daily_filtered = df_daily_filtered[df_daily_filtered['Season'] == selected_season]

# Year filter
years = ['All'] + sorted(df_daily['Year'].unique().tolist())
selected_year = st.sidebar.selectbox("Select Year", years)

if selected_year != 'All':
    df_daily_filtered = df_daily_filtered[df_daily_filtered['Year'] == selected_year]

st.sidebar.markdown("---")
st.sidebar.info(f"📅 Showing {len(df_daily_filtered)} days of data")

# Key Metrics
st.header("📈 Key Climate Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_temp = df_daily_filtered['Temp_Mean'].mean()
    st.metric("Average Temperature", f"{avg_temp:.1f}°C")

with col2:
    avg_humidity = df_daily_filtered['Humidity'].mean()
    st.metric("Average Humidity", f"{avg_humidity:.2f}")

with col3:
    avg_pressure = df_daily_filtered['Pressure'].mean()
    st.metric("Average Pressure", f"{avg_pressure:.1f} mb")

with col4:
    avg_wind = df_daily_filtered['Wind_Speed'].mean()
    st.metric("Average Wind Speed", f"{avg_wind:.1f} km/h")

st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Time Series", "🌡️ Temperature Analysis", "📉 Correlations", "📋 Data Explorer", "📖 Story Mode"])

# Tab 1: Time Series
with tab1:
    st.header("Time Series Analysis")
    
    # Variable selector
    variable_map = {
        'Temperature': 'Temp_Mean',
        'Humidity': 'Humidity',
        'Pressure': 'Pressure',
        'Wind Speed': 'Wind_Speed',
        'Visibility': 'Visibility'
    }
    
    selected_var = st.selectbox("Select Variable", list(variable_map.keys()))
    var_col = variable_map[selected_var]
    
    # Plotly time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily_filtered['Date'],
        y=df_daily_filtered[var_col],
        mode='lines',
        name=selected_var,
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.update_layout(
        title=f"{selected_var} Over Time",
        xaxis_title="Date",
        yaxis_title=selected_var,
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-variable comparison
    st.subheader("Multi-Variable Comparison")
    selected_vars = st.multiselect(
        "Select variables to compare",
        list(variable_map.keys()),
        default=['Temperature', 'Humidity']
    )
    
    if selected_vars:
        fig2 = go.Figure()
        for var in selected_vars:
            col = variable_map[var]
            # Normalize data for comparison
            normalized = (df_daily_filtered[col] - df_daily_filtered[col].min()) / \
                        (df_daily_filtered[col].max() - df_daily_filtered[col].min())
            fig2.add_trace(go.Scatter(
                x=df_daily_filtered['Date'],
                y=normalized,
                mode='lines',
                name=var
            ))
        
        fig2.update_layout(
            title="Normalized Multi-Variable Comparison",
            xaxis_title="Date",
            yaxis_title="Normalized Value (0-1)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Temperature Analysis
with tab2:
    st.header("Temperature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal boxplot
        st.subheader("Temperature by Season")
        fig3 = px.box(
            df_daily_filtered,
            x='Season',
            y='Temp_Mean',
            color='Season',
            category_orders={'Season': ['Winter', 'Spring', 'Summer', 'Fall']},
            title="Temperature Distribution by Season"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Yearly boxplot
        st.subheader("Temperature by Year")
        fig4 = px.box(
            df_daily_filtered,
            x='Year',
            y='Temp_Mean',
            color='Year',
            title="Temperature Distribution by Year"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Temperature heatmap by month and year
    st.subheader("Temperature Heatmap (Month vs Year)")
    pivot_data = df_daily_filtered.groupby(['Year', 'Month'])['Temp_Mean'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='Month', columns='Year', values='Temp_Mean')
    
    fig5 = px.imshow(
        pivot_table,
        labels=dict(x="Year", y="Month", color="Temperature (°C)"),
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale='RdYlBu_r',
        aspect="auto"
    )
    fig5.update_layout(height=400)
    st.plotly_chart(fig5, use_container_width=True)

# Tab 3: Correlations
with tab3:
    st.header("Variable Correlations")
    
    # Correlation matrix
    corr_vars = ['Temp_Mean', 'Humidity', 'Wind_Speed', 'Pressure', 'Visibility']
    corr_matrix = df_daily_filtered[corr_vars].corr()
    
    fig6 = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_vars,
        y=corr_vars,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        text_auto='.2f'
    )
    fig6.update_layout(
        title="Correlation Matrix",
        height=500
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Scatter plot with variable selection
    st.subheader("Scatter Plot Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("X-axis variable", list(variable_map.keys()), index=0)
    with col2:
        y_var = st.selectbox("Y-axis variable", list(variable_map.keys()), index=1)
    
    x_col = variable_map[x_var]
    y_col = variable_map[y_var]
    
    fig7 = px.scatter(
        df_daily_filtered,
        x=x_col,
        y=y_col,
        color='Season',
        title=f"{x_var} vs {y_var}",
        trendline="ols",
        opacity=0.6
    )
    st.plotly_chart(fig7, use_container_width=True)

# Tab 4: Data Explorer
with tab4:
    st.header("Data Explorer")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df_daily_filtered[corr_vars].describe(), use_container_width=True)
    
    # Raw data view
    st.subheader("Raw Data")
    st.dataframe(df_daily_filtered.head(100), use_container_width=True)
    
    # Download button
    csv = df_daily_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"climate_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Tab 5: Story Mode
with tab5:
    st.header("📖 Climate Story: A Decade of Weather Patterns (2006-2016)")
    
    # Story navigation
    story_section = st.selectbox(
        "Choose a story chapter:",
        [
            "🌍 The Big Picture: Our Climate Dataset",
            "🌡️ Chapter 1: The Temperature Tale",
            "💧 Chapter 2: The Humidity Chronicles", 
            "🌪️ Chapter 3: Pressure & Wind Dynamics",
            "🔗 Chapter 4: How Everything Connects",
            "📊 Chapter 5: Seasonal Rhythms",
            "🎯 Key Insights & What They Mean"
        ]
    )
    
    if story_section == "🌍 The Big Picture: Our Climate Dataset":
        st.markdown("---")
        st.subheader("🌍 Welcome to Our Climate Journey")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            **What you're looking at:** A decade of hourly weather observations from 2006 to 2016, 
            containing over 96,000 data points that tell the story of our changing climate.
            
            **Why this matters:** Climate data isn't just numbers—it's the story of our planet's 
            breathing patterns, seasonal rhythms, and the subtle changes that affect all life on Earth.
            
            **The journey ahead:** We'll explore how temperature dances with humidity, how pressure 
            systems drive wind patterns, and how these variables create the weather you experience 
            every day.
            """)
            
        with col2:
            st.info(f"""
            **Dataset at a glance:**
            - 📅 **Time span:** {(df_daily['Date'].max() - df_daily['Date'].min()).days:,} days
            - 🌡️ **Temperature range:** {df_daily['Temp_Min'].min():.1f}°C to {df_daily['Temp_Max'].max():.1f}°C
            - 💨 **Max wind speed:** {df_daily['Wind_Speed'].max():.1f} km/h
            - 👁️ **Visibility range:** {df_daily['Visibility'].min():.1f} to {df_daily['Visibility'].max():.1f} km
            """)
        
        st.markdown("---")
        st.markdown("**🎯 What makes this analysis special:**")
        st.markdown("""
        - **Multi-scale perspective:** We analyze the same data at hourly, daily, and monthly scales
        - **Pattern recognition:** We don't just show you the data—we reveal the hidden patterns
        - **Real-world connections:** Every chart connects back to weather phenomena you've experienced
        - **Interactive exploration:** You can filter and explore the data to test your own hypotheses
        """)
    
    elif story_section == "🌡️ Chapter 1: The Temperature Tale":
        st.markdown("---")
        st.subheader("🌡️ The Temperature Story: Earth's Thermal Heartbeat")
        
        # Temperature trend visualization with story
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_monthly_filtered['Date'],
            y=df_monthly_filtered['Temp_Mean'],
            mode='lines+markers',
            name='Monthly Average',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="🌡️ Temperature's Decade-Long Dance",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Story narrative
        temp_mean = df_daily_filtered['Temp_Mean'].mean()
        temp_trend = np.polyfit(range(len(df_monthly_filtered)), df_monthly_filtered['Temp_Mean'], 1)[0] * 12
        
        st.markdown(f"""
        **📖 What the temperature is telling us:**
        
        The red line above isn't just data—it's Earth's thermal signature over a decade. Here's what it reveals:
        
        **🔥 The Baseline:** Our average temperature sits at **{temp_mean:.1f}°C**, but this number hides 
        a fascinating story of seasonal swings and gradual changes.
        
        **📈 The Trend:** Over this decade, we see a {'warming' if temp_trend > 0 else 'cooling'} trend of 
        **{abs(temp_trend):.3f}°C per year**. While this might seem small, in climate terms, 
        this is {'significant' if abs(temp_trend) > 0.1 else 'modest but measurable'}.
        
        **🌊 The Rhythm:** Notice the beautiful wave pattern? That's our planet's seasonal breathing—
        the annual cycle that governs growing seasons, energy consumption, and ecosystem timing.
        """)
        
        # Seasonal breakdown
        st.markdown("**🌱 Breaking down the seasons:**")
        seasonal_temps = df_daily_filtered.groupby('Season')['Temp_Mean'].agg(['mean', 'std']).round(2)
        
        col1, col2, col3, col4 = st.columns(4)
        seasons_info = {
            'Winter': ('❄️', '#3498db'),
            'Spring': ('🌸', '#2ecc71'), 
            'Summer': ('☀️', '#f39c12'),
            'Fall': ('🍂', '#e67e22')
        }
        
        for i, (season, (emoji, color)) in enumerate(seasons_info.items()):
            with [col1, col2, col3, col4][i]:
                if season in seasonal_temps.index:
                    mean_temp = seasonal_temps.loc[season, 'mean']
                    std_temp = seasonal_temps.loc[season, 'std']
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; border-radius: 10px; background-color: {color}20; border: 2px solid {color}40;'>
                        <h3>{emoji} {season}</h3>
                        <p><strong>{mean_temp}°C</strong><br/>
                        <small>±{std_temp}°C variation</small></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 **Climate Insight:** The difference between summer and winter averages shows the strength of seasonal forcing in this region. Larger differences indicate more continental climate patterns, while smaller differences suggest oceanic influences.")
    
    elif story_section == "💧 Chapter 2: The Humidity Chronicles":
        st.markdown("---")
        st.subheader("💧 Humidity: The Invisible Climate Player")
        
        # Humidity analysis
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_daily_filtered['Date'],
            y=df_daily_filtered['Humidity'],
            mode='lines',
            name='Daily Humidity',
            line=dict(color='#3498db', width=1),
            opacity=0.7
        ))
        
        # Add monthly trend
        monthly_humidity = df_daily_filtered.groupby(df_daily_filtered['Date'].dt.to_period('M'))['Humidity'].mean()
        fig.add_trace(go.Scatter(
            x=monthly_humidity.index.to_timestamp(),
            y=monthly_humidity.values,
            mode='lines',
            name='Monthly Average',
            line=dict(color='#2c3e50', width=3)
        ))
        
        fig.update_layout(
            title="💧 Humidity's Hidden Patterns",
            xaxis_title="Date",
            yaxis_title="Humidity (0-1 scale)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        humidity_mean = df_daily_filtered['Humidity'].mean()
        humidity_seasonal = df_daily_filtered.groupby('Season')['Humidity'].mean()
        
        st.markdown(f"""
        **💧 The Humidity Story:**
        
        Humidity is climate's invisible hand—you can't see it, but you definitely feel it. Our data reveals:
        
        **🌊 The Average:** Humidity hovers around **{humidity_mean:.2f}** (on a 0-1 scale), meaning the air 
        typically holds about **{humidity_mean*100:.0f}%** of the moisture it could at that temperature.
        
        **🔄 Seasonal Moisture Cycles:** 
        - **Summer** brings the highest humidity (**{humidity_seasonal.get('Summer', 0):.2f}**) as warm air holds more moisture
        - **Winter** shows the lowest humidity (**{humidity_seasonal.get('Winter', 0):.2f}**) as cold air can't hold as much water vapor
        
        **⚡ Weather Events:** Those sharp spikes you see? They often mark the passage of weather fronts, 
        storms, or precipitation events—moments when the atmosphere suddenly becomes saturated with moisture.
        """)
        
        # Humidity vs Temperature relationship
        st.markdown("**🌡️💧 The Temperature-Humidity Dance:**")
        
        fig2 = px.scatter(
            df_daily_filtered.sample(1000),  # Sample for performance
            x='Temp_Mean',
            y='Humidity',
            color='Season',
            title="How Temperature and Humidity Move Together",
            labels={'Temp_Mean': 'Temperature (°C)', 'Humidity': 'Humidity (0-1)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        temp_humidity_corr = df_daily_filtered['Temp_Mean'].corr(df_daily_filtered['Humidity'])
        st.markdown(f"""
        **🔗 The Connection:** Temperature and humidity show a **{temp_humidity_corr:.2f}** correlation. 
        {'This negative relationship makes sense—as temperature rises, relative humidity often falls (unless moisture is being added to the air).' if temp_humidity_corr < 0 else 'This positive relationship suggests that warmer periods often coincide with more moisture in the air.'}
        """)
    
    elif story_section == "🌪️ Chapter 3: Pressure & Wind Dynamics":
        st.markdown("---")
        st.subheader("🌪️ Pressure & Wind: The Atmosphere's Engine")
        
        # Pressure and wind analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_daily_filtered['Date'],
                y=df_daily_filtered['Pressure'],
                mode='lines',
                name='Daily Pressure',
                line=dict(color='#9b59b6', width=1)
            ))
            fig.update_layout(
                title="🌪️ Atmospheric Pressure Patterns",
                xaxis_title="Date",
                yaxis_title="Pressure (mb)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_daily_filtered['Date'],
                y=df_daily_filtered['Wind_Speed'],
                mode='lines',
                name='Daily Wind Speed',
                line=dict(color='#1abc9c', width=1)
            ))
            fig.update_layout(
                title="💨 Wind Speed Variations",
                xaxis_title="Date", 
                yaxis_title="Wind Speed (km/h)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Pressure-Wind relationship
        st.markdown("**🌪️ The Pressure-Wind Connection:**")
        
        fig3 = px.scatter(
            df_daily_filtered.sample(1000),
            x='Pressure',
            y='Wind_Speed',
            color='Season',
            title="How Pressure Drives Wind Patterns",
            labels={'Pressure': 'Atmospheric Pressure (mb)', 'Wind_Speed': 'Wind Speed (km/h)'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        pressure_wind_corr = df_daily_filtered['Pressure'].corr(df_daily_filtered['Wind_Speed'])
        pressure_mean = df_daily_filtered['Pressure'].mean()
        wind_mean = df_daily_filtered['Wind_Speed'].mean()
        
        st.markdown(f"""
        **🌪️ What Pressure and Wind Tell Us:**
        
        **⚖️ Pressure Baseline:** Our average atmospheric pressure is **{pressure_mean:.1f} mb**—this is 
        the weight of the entire atmosphere pressing down on us!
        
        **💨 Wind Patterns:** Average wind speed is **{wind_mean:.1f} km/h**, but the story is in the extremes. 
        High wind events often coincide with pressure changes.
        
        **🔗 The Physics:** Pressure and wind show a **{pressure_wind_corr:.2f}** correlation. 
        {'This negative relationship is classic meteorology—low pressure systems create pressure gradients that drive strong winds.' if pressure_wind_corr < 0 else 'This relationship shows how pressure variations drive atmospheric motion.'}
        
        **🌀 Storm Signatures:** Look for periods where pressure drops sharply while wind speed spikes—
        these are often signatures of passing storm systems or weather fronts.
        """)
        
        # Extreme events
        low_pressure_days = df_daily_filtered[df_daily_filtered['Pressure'] < df_daily_filtered['Pressure'].quantile(0.1)]
        high_wind_days = df_daily_filtered[df_daily_filtered['Wind_Speed'] > df_daily_filtered['Wind_Speed'].quantile(0.9)]
        
        st.info(f"""
        **⚡ Extreme Weather Signatures:**
        - **Low pressure events:** {len(low_pressure_days)} days with pressure below {df_daily_filtered['Pressure'].quantile(0.1):.1f} mb
        - **High wind events:** {len(high_wind_days)} days with winds above {df_daily_filtered['Wind_Speed'].quantile(0.9):.1f} km/h
        - **Combined events:** {len(set(low_pressure_days.index) & set(high_wind_days.index))} days with both low pressure AND high winds
        """)
    
    elif story_section == "🔗 Chapter 4: How Everything Connects":
        st.markdown("---")
        st.subheader("🔗 The Web of Climate: How Variables Dance Together")
        
        # Correlation matrix with story
        corr_vars = ['Temp_Mean', 'Humidity', 'Wind_Speed', 'Pressure', 'Visibility']
        corr_matrix = df_daily_filtered[corr_vars].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation Strength"),
            x=['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Visibility'],
            y=['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Visibility'],
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            text_auto='.2f',
            title="🔗 The Climate Connection Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **🔗 Reading the Connection Map:**
        
        This heatmap reveals the hidden relationships between climate variables. Here's how to read it:
        - **Red colors** = Positive correlation (when one goes up, the other tends to go up)
        - **Blue colors** = Negative correlation (when one goes up, the other tends to go down)  
        - **White/neutral** = Little to no relationship
        """)
        
        # Strongest relationships
        strongest_corrs = []
        for i in range(len(corr_vars)):
            for j in range(i+1, len(corr_vars)):
                corr_val = corr_matrix.iloc[i, j]
                strongest_corrs.append((corr_vars[i], corr_vars[j], corr_val))
        
        strongest_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        st.markdown("**🏆 The Strongest Climate Relationships:**")
        
        for i, (var1, var2, corr_val) in enumerate(strongest_corrs[:3]):
            var1_name = var1.replace('_', ' ').replace('Temp Mean', 'Temperature')
            var2_name = var2.replace('_', ' ').replace('Temp Mean', 'Temperature')
            
            relationship = "positively" if corr_val > 0 else "negatively"
            strength = "strongly" if abs(corr_val) > 0.5 else "moderately" if abs(corr_val) > 0.3 else "weakly"
            
            st.markdown(f"""
            **{i+1}. {var1_name} ↔ {var2_name}:** {strength} {relationship} correlated ({corr_val:+.2f})
            """)
        
        # Interactive scatter plot
        st.markdown("**🔍 Explore Relationships Yourself:**")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Choose X-axis variable:", 
                               ['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Visibility'])
        with col2:
            y_var = st.selectbox("Choose Y-axis variable:", 
                               ['Humidity', 'Temperature', 'Wind Speed', 'Pressure', 'Visibility'])
        
        if x_var != y_var:
            x_col = {'Temperature': 'Temp_Mean', 'Humidity': 'Humidity', 
                    'Wind Speed': 'Wind_Speed', 'Pressure': 'Pressure', 'Visibility': 'Visibility'}[x_var]
            y_col = {'Temperature': 'Temp_Mean', 'Humidity': 'Humidity', 
                    'Wind Speed': 'Wind_Speed', 'Pressure': 'Pressure', 'Visibility': 'Visibility'}[y_var]
            
            fig_scatter = px.scatter(
                df_daily_filtered.sample(1000),
                x=x_col,
                y=y_col,
                color='Season',
                title=f"{x_var} vs {y_var} Relationship",
                trendline="ols",
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            correlation = df_daily_filtered[x_col].corr(df_daily_filtered[y_col])
            st.info(f"**Correlation between {x_var} and {y_var}: {correlation:+.3f}**")
    
    elif story_section == "📊 Chapter 5: Seasonal Rhythms":
        st.markdown("---")
        st.subheader("📊 Nature's Calendar: The Rhythm of Seasons")
        
        st.markdown("""
        **🌍 The Seasonal Story:**
        
        Seasons aren't just calendar dates—they're the fundamental rhythm that drives most climate patterns. 
        Let's explore how each season leaves its unique signature in our data.
        """)
        
        # Seasonal comparison across all variables
        seasonal_stats = df_daily_filtered.groupby('Season')[['Temp_Mean', 'Humidity', 'Wind_Speed', 'Pressure', 'Visibility']].agg(['mean', 'std']).round(2)
        
        # Create seasonal profiles
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        season_colors = {'Winter': '#3498db', 'Spring': '#2ecc71', 'Summer': '#f39c12', 'Fall': '#e67e22'}
        season_emojis = {'Winter': '❄️', 'Spring': '🌸', 'Summer': '☀️', 'Fall': '🍂'}
        
        for season in seasons:
            if season in seasonal_stats.index:
                st.markdown(f"### {season_emojis[season]} {season} Profile")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    temp_mean = seasonal_stats.loc[season, ('Temp_Mean', 'mean')]
                    temp_std = seasonal_stats.loc[season, ('Temp_Mean', 'std')]
                    st.metric("🌡️ Temperature", f"{temp_mean}°C", f"±{temp_std}°C")
                
                with col2:
                    hum_mean = seasonal_stats.loc[season, ('Humidity', 'mean')]
                    st.metric("💧 Humidity", f"{hum_mean:.2f}", f"{hum_mean*100:.0f}%")
                
                with col3:
                    wind_mean = seasonal_stats.loc[season, ('Wind_Speed', 'mean')]
                    st.metric("💨 Wind Speed", f"{wind_mean:.1f} km/h")
                
                with col4:
                    pressure_mean = seasonal_stats.loc[season, ('Pressure', 'mean')]
                    st.metric("🌪️ Pressure", f"{pressure_mean:.0f} mb")
                
                with col5:
                    vis_mean = seasonal_stats.loc[season, ('Visibility', 'mean')]
                    st.metric("👁️ Visibility", f"{vis_mean:.1f} km")
                
                # Seasonal insights
                season_insights = {
                    'Winter': "❄️ **Winter's signature:** Cold temperatures, variable humidity, often higher pressure systems bringing clear but cold conditions.",
                    'Spring': "🌸 **Spring's transition:** Warming temperatures, increasing humidity as the atmosphere can hold more moisture, variable weather patterns.",
                    'Summer': "☀️ **Summer's intensity:** Peak temperatures, highest humidity, often lower pressure with more convective activity.",
                    'Fall': "🍂 **Fall's balance:** Cooling temperatures, decreasing humidity, transitional weather patterns as systems shift."
                }
                
                st.markdown(season_insights.get(season, ""))
                st.markdown("---")
        
        # Seasonal variability chart
        st.markdown("**📈 Seasonal Variability Patterns:**")
        
        fig = go.Figure()
        
        variables = ['Temp_Mean', 'Humidity', 'Wind_Speed', 'Pressure', 'Visibility']
        var_names = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Visibility']
        
        for i, (var, name) in enumerate(zip(variables, var_names)):
            # Normalize each variable to 0-1 scale for comparison
            seasonal_means = df_daily_filtered.groupby('Season')[var].mean()
            seasonal_means_norm = (seasonal_means - seasonal_means.min()) / (seasonal_means.max() - seasonal_means.min())
            
            fig.add_trace(go.Scatterpolar(
                r=seasonal_means_norm.reindex(seasons).values,
                theta=seasons,
                fill='toself',
                name=name,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🌍 Seasonal Climate Fingerprints (Normalized)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **💡 Reading the Seasonal Fingerprints:**
        Each colored area shows how a climate variable changes across seasons (normalized to 0-1 scale). 
        Larger areas indicate more seasonal variation, while smaller areas suggest more stable year-round patterns.
        """)
    
    elif story_section == "🎯 Key Insights & What They Mean":
        st.markdown("---")
        st.subheader("🎯 The Big Picture: What Our Climate Data Reveals")
        
        st.markdown("""
        After exploring a decade of climate data, here are the key insights that emerge from the patterns:
        """)
        
        # Key insights with supporting data
        insights = [
            {
                "title": "🌊 Seasonal Cycles Dominate Climate Variability",
                "insight": f"Seasonal patterns explain most of the variation in our climate data. The temperature difference between summer and winter averages is {df_daily_filtered.groupby('Season')['Temp_Mean'].mean().max() - df_daily_filtered.groupby('Season')['Temp_Mean'].mean().min():.1f}°C, showing strong seasonal forcing.",
                "implication": "This confirms that seasonal planning (energy use, agriculture, tourism) should be the primary consideration for climate adaptation strategies."
            },
            {
                "title": "🔗 Climate Variables Are Interconnected",
                "insight": f"Temperature and humidity show a {df_daily_filtered['Temp_Mean'].corr(df_daily_filtered['Humidity']):.2f} correlation, while pressure and wind speed correlate at {df_daily_filtered['Pressure'].corr(df_daily_filtered['Wind_Speed']):.2f}. These aren't independent systems—they're part of an integrated climate machine.",
                "implication": "Weather prediction and climate modeling must consider these relationships. Changes in one variable will cascade through the entire system."
            },
            {
                "title": "⚡ Extreme Events Leave Clear Signatures",
                "insight": f"We identified {len(df_daily_filtered[df_daily_filtered['Wind_Speed'] > df_daily_filtered['Wind_Speed'].quantile(0.9)])} high-wind events and {len(df_daily_filtered[df_daily_filtered['Pressure'] < df_daily_filtered['Pressure'].quantile(0.1)])} low-pressure events. Many coincide, showing storm system signatures.",
                "implication": "Early warning systems can use multi-variable thresholds to better predict severe weather events."
            },
            {
                "title": "📈 Long-term Trends Emerge from Noise",
                "insight": f"While daily weather is noisy, monthly aggregation reveals clear patterns. Our data shows a temperature trend of {np.polyfit(range(len(df_monthly_filtered)), df_monthly_filtered['Temp_Mean'], 1)[0] * 12:.3f}°C per year over the decade.",
                "implication": "Climate analysis requires the right time scale. Daily data shows weather; monthly data reveals climate patterns."
            }
        ]
        
        for i, insight_data in enumerate(insights, 1):
            with st.expander(f"**Insight {i}: {insight_data['title']}**", expanded=True):
                st.markdown(f"**📊 What we found:** {insight_data['insight']}")
                st.markdown(f"**🎯 What it means:** {insight_data['implication']}")
        
        st.markdown("---")
        
        # Summary statistics dashboard
        st.markdown("### 📊 Climate Summary Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌡️ Average Temperature", 
                f"{df_daily_filtered['Temp_Mean'].mean():.1f}°C",
                f"Range: {df_daily_filtered['Temp_Min'].min():.1f}° to {df_daily_filtered['Temp_Max'].max():.1f}°C"
            )
        
        with col2:
            st.metric(
                "💧 Average Humidity", 
                f"{df_daily_filtered['Humidity'].mean():.2f}",
                f"{df_daily_filtered['Humidity'].mean()*100:.0f}% relative humidity"
            )
        
        with col3:
            st.metric(
                "💨 Average Wind Speed", 
                f"{df_daily_filtered['Wind_Speed'].mean():.1f} km/h",
                f"Max: {df_daily_filtered['Wind_Speed'].max():.1f} km/h"
            )
        
        with col4:
            st.metric(
                "🌪️ Average Pressure", 
                f"{df_daily_filtered['Pressure'].mean():.0f} mb",
                f"Range: {df_daily_filtered['Pressure'].min():.0f}-{df_daily_filtered['Pressure'].max():.0f} mb"
            )
        
        st.markdown("---")
        
        # Call to action
        st.success("""
        **🌍 The Climate Story Continues...**
        
        This analysis represents just one decade of Earth's climate story. The patterns we've discovered—
        seasonal rhythms, variable relationships, and extreme event signatures—provide a foundation 
        for understanding how our climate system works.
        
        **What's next?** Use the other tabs to explore the data yourself, test hypotheses, and discover 
        your own insights. Climate science is about finding patterns in complexity, and every 
        exploration reveals something new about our planet's remarkable atmospheric system.
        """)
        
        # Data quality note
        st.info("""
        **📋 About this analysis:** This story is based on NOAA weather data from 2006-2016, 
        carefully cleaned and aggregated to reveal meaningful patterns. All processing steps 
        are documented and reproducible. The insights reflect this specific dataset and time period.
        """)
    
    st.markdown("---")
    st.markdown("*Navigate between story chapters using the dropdown above to explore different aspects of our climate data.*")
  
# ...existing code...
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🌍 Global Weather & Climate Analysis Dashboard | Data Source: NOAA (2006-2016)</p>
    </div>
    """, unsafe_allow_html=True)
