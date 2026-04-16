"""
🔥 California Wildfire Risk Dashboard

A Streamlit app for visualizing wildfire predictions.

To run locally:
    streamlit run app.py

To deploy:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect your repo
    4. Deploy!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ========================================
# PAGE CONFIG (must be first Streamlit command)
# ========================================

st.set_page_config(
    page_title="Wildfire Risk Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# LOAD DATA
# ========================================

@st.cache_data
def load_data():
    """Load predictions data."""
    try:
        # Load your predictions CSV
        df = pd.read_csv('data/predictions_for_dashboard.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Make sure 'data/predictions_for_dashboard.csv' exists.")
        st.stop()

# Load data
predictions = load_data()

# ========================================
# SIDEBAR
# ========================================

st.sidebar.title("🔥 Fire Risk Monitor")
st.sidebar.markdown("---")

# Date selector
available_dates = sorted(predictions['date'].unique())
selected_date = st.sidebar.selectbox(
    "📅 Select Date",
    available_dates,
    format_func=lambda x: pd.to_datetime(x).strftime('%B %d, %Y')
)

# Threshold slider
threshold = st.sidebar.slider(
    "⚖️ Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Adjust threshold for different use cases"
)

# Threshold guidance
st.sidebar.markdown("### 💡 Threshold Guide")
if threshold <= 0.2:
    st.sidebar.warning("🚨 **Evacuation Mode**\nHigh recall, more false alarms")
elif threshold <= 0.5:
    st.sidebar.success("⚖️ **Balanced Mode**\nGood precision-recall tradeoff")
else:
    st.sidebar.info("🎯 **High Confidence**\nFew false alarms, lower recall")

st.sidebar.markdown("---")

# Model info
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
**Algorithm**: LightGBM  
**AUPRC**: 0.311  
**Improvement**: 54× vs random  

**Data Sources**:  
• 🛰️ MODIS (satellite)  
• 🌡️ ERA5 (weather)  
• ⛰️ SRTM (terrain)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**GitHub**: [View Code](https://github.com/yourusername/wildfire-prediction)")

# ========================================
# MAIN CONTENT
# ========================================

# Title
st.markdown("# 🔥 California Wildfire Risk Prediction System")
st.markdown("### 7-day ahead predictions using satellite imagery and machine learning")
st.markdown("---")

# Filter data for selected date
daily_data = predictions[predictions['date'] == selected_date].copy()
daily_data['actual_fire'] = daily_data['actual_fire'].astype(bool)
#added above line for boolean data
# Calculate predictions based on threshold
daily_data['predicted_fire'] = daily_data['fire_risk'] >= threshold

# Calculate metrics
tp = ((daily_data['predicted_fire']) & (daily_data['actual_fire'])).sum()
fp = ((daily_data['predicted_fire']) & (~daily_data['actual_fire'])).sum()
fn = ((~daily_data['predicted_fire']) & (daily_data['actual_fire'])).sum()
tn = ((~daily_data['predicted_fire']) & (~daily_data['actual_fire'])).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# ========================================
# METRICS ROW
# ========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk_count = (daily_data['fire_risk'] > threshold).sum()
    st.metric(
        "🚨 High Risk Areas",
        f"{high_risk_count:,}",
        help=f"Cells with risk > {threshold:.0%}"
    )

with col2:
    st.metric(
        "🎯 Precision",
        f"{precision:.1%}",
        help="% of predictions that are correct"
    )

with col3:
    st.metric(
        "📊 Recall",
        f"{recall:.1%}",
        help="% of actual fires detected"
    )

with col4:
    actual_fires = daily_data['actual_fire'].sum()
    st.metric(
        "🔥 Actual Fires",
        f"{int(actual_fires)}",
        help="Fires that occurred on this date"
    )

st.markdown("---")

# ========================================
# TABS
# ========================================

tab1, tab2, tab3 = st.tabs(["🗺️ Risk Map", "📈 Time Series", "🎯 Performance"])

# ----------------------------------------
# TAB 1: RISK MAP
# ----------------------------------------

with tab1:
    st.subheader(f"Fire Risk Map - {pd.to_datetime(selected_date).strftime('%B %d, %Y')}")
    
    # Create map
    fig_map = px.scatter_mapbox(
        daily_data,
        lat='lat',
        lon='lon',
        color='fire_risk',
        size='fire_risk',
        hover_data={
            'fire_risk': ':.2%',
            'elevation_m': ':.0f',
            'temp_mean_C': ':.1f',
            'actual_fire': True,
            'lat': False,
            'lon': False
        },
        color_continuous_scale='YlOrRd',
        range_color=[0, 1],
        size_max=15,
        zoom=5.5,
        height=600,
        labels={'fire_risk': 'Fire Risk Probability'}
    )
    
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=37, lon=-120)),
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Risk distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Risk Distribution")
        
        risk_bins = pd.cut(
            daily_data['fire_risk'], 
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low\n(0-20%)', 'Low\n(20-40%)', 'Moderate\n(40-60%)', 
                    'High\n(60-80%)', 'Extreme\n(80-100%)']
        )
        risk_counts = risk_bins.value_counts().sort_index()
        
        fig_dist = px.bar(
            x=risk_counts.index.astype(str),
            y=risk_counts.values,
            labels={'x': 'Risk Level', 'y': 'Number of Cells'},
            color=risk_counts.values,
            color_continuous_scale='YlOrRd'
        )
        fig_dist.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Top 10 Highest Risk Locations")
        
        top_risk = daily_data.nlargest(10, 'fire_risk')[
            ['fire_risk', 'elevation_m', 'temp_mean_C', 'actual_fire']
        ].copy()
        
        top_risk['fire_risk'] = top_risk['fire_risk'].apply(lambda x: f"{x:.1%}")
        top_risk['elevation_m'] = top_risk['elevation_m'].round(0).astype(int)
        top_risk['temp_mean_C'] = top_risk['temp_mean_C'].round(1)
        top_risk['actual_fire'] = top_risk['actual_fire'].map({True: '🔥 Yes', False: '✅ No'})
        
        top_risk.columns = ['Risk', 'Elevation (m)', 'Temp (°C)', 'Fire?']
        
        st.dataframe(top_risk, hide_index=True, use_container_width=True)

# ----------------------------------------
# TAB 2: TIME SERIES
# ----------------------------------------
# Convert to python datetime (just to be safe)
selected_dt = pd.to_datetime(selected_date).to_pydatetime()

with tab2:
    st.subheader("Risk Evolution Over Time")
    
    # Calculate daily statistics
    daily_stats = predictions.groupby('date').agg({
        'fire_risk': ['mean', 'max'],
        'actual_fire': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'avg_risk', 'max_risk', 'actual_fires']
    
    # Create time series plot
    fig_ts = go.Figure()
    
    # Average risk
    fig_ts.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['avg_risk'],
        name='Average Risk',
        line=dict(color='#FFA500', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))
    
    # Maximum risk
    fig_ts.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['max_risk'],
        name='Maximum Risk',
        line=dict(color='#FF4500', width=2, dash='dash')
    ))
    
    # Actual fires (on secondary axis)
    fig_ts.add_trace(go.Bar(
        x=daily_stats['date'],
        y=daily_stats['actual_fires'],
        name='Actual Fires',
        marker_color='#8B0000',
        opacity=0.6,
        yaxis='y2'
    ))
    
    # Highlight selected date
    # Add vertical line
    fig_ts.add_shape(
     type="line",
     x0=selected_dt,
     x1=selected_dt,
     y0=0,
     y1=1,
     xref="x",
     yref="paper",
     line=dict(color="green", dash="dash", width=2),
    )  
    
    fig_ts.add_annotation(
     x=selected_dt,
     y=1,
     xref="x",
     yref="paper",
     text="Selected Date",
     showarrow=False,
     yshift=10,
     font=dict(color="green")
    )

    fig_ts.update_layout(
        height=400,
        xaxis_title='Date',
        yaxis_title='Fire Risk Probability',
        yaxis2=dict(
            title='Number of Fires',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Daily summary
    col1, col2, col3 = st.columns(3)
    
    selected_stats = daily_stats[daily_stats['date'] == selected_date].iloc[0]
    
    with col1:
        st.metric("Average Risk", f"{selected_stats['avg_risk']:.1%}")
    with col2:
        st.metric("Peak Risk", f"{selected_stats['max_risk']:.1%}")
    with col3:
        st.metric("Actual Fires", int(selected_stats['actual_fires']))

# ----------------------------------------
# TAB 3: PERFORMANCE
# ----------------------------------------

with tab3:
    st.subheader("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        
        # Confusion matrix
        cm_data = np.array([[tn, fp], [fn, tp]])
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted No Fire', 'Predicted Fire'],
            y=['Actual No Fire', 'Actual Fire'],
            text=cm_data,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 18},
            colorscale='Blues',
            showscale=False,
            hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig_cm.update_layout(
            height=350,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Metrics breakdown
        st.info(f"""
        **Confusion Matrix Breakdown:**
        - ✅ True Positives: {tp} (fires correctly predicted)
        - ⚠️ False Positives: {fp} (false alarms)
        - ❌ False Negatives: {fn} (missed fires)
        - ✅ True Negatives: {tn} (correct no-fire predictions)
        """)
    
    with col2:
        st.markdown("#### Threshold Sensitivity")
        
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        metrics = []
        
        for t in thresholds:
            daily_data['pred_t'] = daily_data['fire_risk'] >= t
            tp_t = ((daily_data['pred_t']) & (daily_data['actual_fire'])).sum()
            fp_t = ((daily_data['pred_t']) & (~daily_data['actual_fire'])).sum()
            fn_t = ((~daily_data['pred_t']) & (daily_data['actual_fire'])).sum()
            
            prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            
            metrics.append({'threshold': t, 'precision': prec, 'recall': rec})
        
        metrics_df = pd.DataFrame(metrics)
        
        # Plot
        fig_thresh = go.Figure()
        
        fig_thresh.add_trace(go.Scatter(
            x=metrics_df['threshold'],
            y=metrics_df['precision'],
            name='Precision',
            line=dict(color='#4CAF50', width=3),
            mode='lines+markers'
        ))
        
        fig_thresh.add_trace(go.Scatter(
            x=metrics_df['threshold'],
            y=metrics_df['recall'],
            name='Recall',
            line=dict(color='#2196F3', width=3),
            mode='lines+markers'
        ))
        
        # Highlight current threshold
        fig_thresh.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current ({threshold:.1f})"
        )
        
        fig_thresh.update_layout(
            height=350,
            xaxis_title='Threshold',
            yaxis_title='Score',
            yaxis_range=[0, 1],
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_thresh, use_container_width=True)
        
        # Recommendations
        st.success(f"""
        **Current Settings (Threshold: {threshold:.1f})**
        - Precision: {precision:.1%}
        - Recall: {recall:.1%}
        - F1 Score: {f1:.1%}
        """)

# ========================================
# FOOTER
# ========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Data Sources**")
    st.caption("🛰️ NASA MODIS")
    st.caption("🌡️ ECMWF ERA5")
    st.caption("⛰️ USGS SRTM")

with col2:
    st.markdown("**🤖 Model**")
    st.caption("Algorithm: LightGBM")
    st.caption("Features: 25 temporal + spatial")
    st.caption("AUPRC: 0.311 (54× vs random)")

with col3:
    st.markdown("**🔗 Links**")
    st.caption("[📄 GitHub Repository](https://github.com/krithi33/wildfire-prediction)")
    st.caption("[💼 LinkedIn](https://linkedin.com/in/krithi-m-s/)")

st.markdown("---")
st.caption("Built with 🔥 for climate impact | © 2025 Your Name")
