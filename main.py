import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the data with proper time handling"""
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Add derived time columns for easier aggregation
    df['Date'] = df['Timestamp'].dt.date
    df['Month'] = df['Timestamp'].dt.to_period('M')
    df['Year'] = df['Timestamp'].dt.year
    
    return df

def aggregate_data(df, time_period):
    """Aggregate data based on selected time period"""
    agg_dict = {
        'Maximum energy deliverable before curtailment (MWh)': 'sum',
        'Energy actually delivered (MWh)': 'sum',
        'Curtailment (MWh)': 'sum',
        'Price ($/MWh) at the nodal level in the Real Time market': 'mean'
    }
    
    if time_period == 'Daily':
        return df.groupby('Date').agg(agg_dict).reset_index()
    elif time_period == 'Monthly':
        monthly = df.groupby('Month').agg(agg_dict).reset_index()
        monthly['Month'] = monthly['Month'].astype(str)
        return monthly
    else:  # Yearly
        return df.groupby('Year').agg(agg_dict).reset_index()

def create_energy_delivery_plot(data, time_col, station_name, time_period):
    """Create an enhanced energy delivery plot with proper spacing"""
    fig = go.Figure()
    
    # Maximum deliverable energy
    fig.add_trace(go.Bar(
        x=data[time_col],
        y=data['Maximum energy deliverable before curtailment (MWh)'],
        name='Maximum Deliverable',
        marker_color='rgba(135, 206, 250, 0.7)',
    ))
    
    # Actually delivered energy
    fig.add_trace(go.Bar(
        x=data[time_col],
        y=data['Energy actually delivered (MWh)'],
        name='Actually Delivered',
        marker_color='rgba(60, 179, 113, 0.7)',
    ))
    
    # Curtailment as a line
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data['Curtailment (MWh)'],
        name='Curtailment',
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f"{station_name} - Energy Delivery Overview ({time_period})",
        barmode='overlay',
        yaxis=dict(
            title="Energy (MWh)",
            side="left"
        ),
        yaxis2=dict(
            title="Curtailment (MWh)",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_price_plot(data, time_col, station_name, time_period):
    """Create an enhanced price analysis plot"""
    fig = go.Figure()
    
    # Average price
    fig.add_trace(go.Scatter(
        x=data[time_col],
        y=data['Price ($/MWh) at the nodal level in the Real Time market'],
        name='Average Price',
        line=dict(color='purple', width=2),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=f"{station_name} - Price Trends ({time_period})",
        xaxis_title=time_period,
        yaxis_title="Price ($/MWh)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    st.title("BESS Data Analyzer")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    battery = st.selectbox("Lithium Ion Battery technology", ["LMO/Graphite", "LFP/Graphite", "LCO/Graphite", "LMO/LTO", "NMC/Graphite", "NCA/Graphite", "Custom"], index=0)

    
    battery_capacity_MW = st.number_input("Battery capacity (MW)", value=4000, min_value=1000, max_value=10000, step=100)
    battery_capacity_h = st.number_input("Battery capacity (h)", value=4, min_value = 1, max_value = 8, step = 1)
    
    battery_capacity = battery_capacity_h * battery_capacity_MW
    
    #for every day select battery_capacity_h minimum hourly prices to charge the battery either 
    #from the grid or the power generation plant. For doing it from grid you need negative prices
    #else you use curtailment from generation plant

    #we discharge at the highest price battery_capacity_h in a day  
    if uploaded_file is not None:
        # Load and preprocess data
        df = load_and_preprocess_data(uploaded_file)
        
        # Station name input
        station_name = st.text_input("Power Station Name", "Valentino")
        
        # Time period selector
        time_period = st.selectbox(
            "Select Time Period",
            ["Daily", "Monthly", "Yearly"]
        )
        
        # Aggregate data based on selected time period
        time_col = 'Date' if time_period == 'Daily' else ('Month' if time_period == 'Monthly' else 'Year')
        agg_data = aggregate_data(df, time_period)
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📝 Data Summary", "📈 Analysis"])
        
        with tab1:
            st.header("Data Visualizations")
            
            # Energy Delivery Plot
            st.subheader("Energy Delivery and Curtailment")
            energy_fig = create_energy_delivery_plot(agg_data, time_col, station_name, time_period)
            st.plotly_chart(energy_fig, use_container_width=True)
            
            # Price Analysis Plot
            st.subheader("Price Analysis")
            price_fig = create_price_plot(agg_data, time_col, station_name, time_period)
            st.plotly_chart(price_fig, use_container_width=True)
        
        with tab2:
            st.header("Data Summary")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Energy Delivered (MWh)",
                    f"{agg_data['Energy actually delivered (MWh)'].sum():,.2f}"
                )
            
            with col2:
                st.metric(
                    "Average Price ($/MWh)",
                    f"{agg_data['Price ($/MWh) at the nodal level in the Real Time market'].mean():,.2f}"
                )
            
            with col3:
                st.metric(
                    "Total Curtailment (MWh)",
                    f"{agg_data['Curtailment (MWh)'].sum():,.2f}"
                )
            
            with col4:
                delivery_ratio = (agg_data['Energy actually delivered (MWh)'].sum() / 
                                agg_data['Maximum energy deliverable before curtailment (MWh)'].sum() * 100)
                st.metric(
                    "Delivery Ratio (%)",
                    f"{delivery_ratio:.1f}"
                )
            
            # Display aggregated data
            st.subheader(f"{time_period} Summary")
            st.dataframe(
                agg_data.style.format({
                    'Maximum energy deliverable before curtailment (MWh)': '{:,.2f}',
                    'Energy actually delivered (MWh)': '{:,.2f}',
                    'Curtailment (MWh)': '{:,.2f}',
                    'Price ($/MWh) at the nodal level in the Real Time market': '{:,.2f}'
                }),
                use_container_width=True
            )
        
        with tab3:
            st.header("Advanced Analysis")
            
            # Delivery efficiency over time
            efficiency_data = agg_data.copy()
            efficiency_data['Delivery Efficiency (%)'] = (
                efficiency_data['Energy actually delivered (MWh)'] / 
                efficiency_data['Maximum energy deliverable before curtailment (MWh)'] * 100
            )
            
            fig = px.line(
                efficiency_data,
                x=time_col,
                y='Delivery Efficiency (%)',
                title=f"Delivery Efficiency Over Time ({time_period})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price distribution
            fig = px.histogram(
                agg_data,
                x='Price ($/MWh) at the nodal level in the Real Time market',
                title=f"Price Distribution ({time_period})",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Power Station Analyzer",
        page_icon="⚡",
        layout="wide"
    )
    main()