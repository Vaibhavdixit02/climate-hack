import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import numpy_financial as npf

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the data with proper time handling"""
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Add derived time columns for easier aggregation
    df['Date'] = df['Timestamp'].dt.date
    df['Month'] = df['Timestamp'].dt.to_period('M')
    df['Year'] = df['Timestamp'].dt.year
    
    return df

def optimize_battery_operations(df, battery_capacity_MW, battery_capacity_h, battery_total_cost = 0.482, efficiency=0.95):
    """
    Optimize battery operations on a daily basis
    
    Parameters:
    - df: DataFrame with hourly data
    - battery_capacity_MW: Battery power capacity in MW
    - battery_capacity_h: Battery duration in hours
    - efficiency: Round-trip efficiency of the battery
    
    Returns:
    - DataFrame with battery operations
    """
    battery_capacity = battery_capacity_MW * battery_capacity_h
    df_with_battery = df.copy()
    
    # Initialize battery operation columns
    df_with_battery['Battery_Charging'] = 0.0
    df_with_battery['Battery_Discharging'] = 0.0
    df_with_battery['Battery_Level'] = 0.0
    df_with_battery['Battery_Revenue'] = 0.0
    df_with_battery['Battery_Cost'] = 0.0
    df_with_battery['Fixed_Cost'] = 0.0
    df_with_battery['Charging_Source'] = ''
    
    # Process each day separately
    for date in df_with_battery['Date'].unique():
        day_data = df_with_battery[df_with_battery['Date'] == date].copy()
        
        # Sort hours by price to find best charging and discharging periods
        charging_hours = day_data.nsmallest(battery_capacity_h, 'Price ($/MWh) at the nodal level in the Real Time market').index
        discharging_hours = day_data.nlargest(battery_capacity_h, 'Price ($/MWh) at the nodal level in the Real Time market').index
        
        for hour_idx in day_data.index:
            price = day_data.loc[hour_idx, 'Price ($/MWh) at the nodal level in the Real Time market']
            curtailment = day_data.loc[hour_idx, 'Curtailment (MWh)']
            
            # Charging logic
            if hour_idx in charging_hours:
                # Determine charging source and amount
                if curtailment > 0:  # Charge from curtailment if available
                    charge_amount = min(curtailment, battery_capacity_MW)
                    df_with_battery.loc[hour_idx, 'Charging_Source'] = 'Curtailment'
                else:
                    charge_amount = battery_capacity_MW
                    df_with_battery.loc[hour_idx, 'Charging_Source'] = 'Grid'

                df_with_battery.loc[hour_idx, 'Battery_Charging'] = charge_amount

            # Discharging logic
            elif hour_idx in discharging_hours and price > 0:
                current_level = df_with_battery.loc[hour_idx-1, 'Battery_Level'] if hour_idx > 0 else 0
                discharge_amount = min(battery_capacity_MW, current_level)
                df_with_battery.loc[hour_idx, 'Battery_Discharging'] = discharge_amount
            
            # Update battery level
            previous_level = df_with_battery.loc[hour_idx-1, 'Battery_Level'] if hour_idx > 0 else 0
            df_with_battery.loc[hour_idx, 'Battery_Level'] = (
                previous_level + 
                df_with_battery.loc[hour_idx, 'Battery_Charging'] * efficiency - 
                df_with_battery.loc[hour_idx, 'Battery_Discharging'] * efficiency
            )
            
            # Calculate revenue
            charging_cost = df_with_battery.loc[hour_idx, 'Battery_Charging'] * price  if df_with_battery.loc[hour_idx, 'Charging_Source'] == 'Grid' else 0 
            discharging_revenue = df_with_battery.loc[hour_idx, 'Battery_Discharging'] * price if df_with_battery.loc[hour_idx, 'Curtailment (MWh)'] == 0 else 0
            fixed_cost = df_with_battery.loc[hour_idx, 'Battery_Charging'] * battery_total_cost
            df_with_battery.loc[hour_idx, 'Battery_Revenue'] = discharging_revenue
            df_with_battery.loc[hour_idx, 'Battery_Cost'] = charging_cost
            df_with_battery.loc[hour_idx, 'Fixed_Cost'] = fixed_cost
    df_with_battery.to_csv('battery_operations.csv')
    return df_with_battery

def aggregate_battery_data(df, period='hour'):
    """
    Aggregate battery data based on specified time period
    
    Parameters:
    - df: DataFrame with battery operation data
    - period: str, one of 'hour', 'day', 'week', 'month'
    
    Returns:
    - Aggregated DataFrame
    """
    # Ensure timestamp is datetime
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Create period mappings
    period_mapping = {
        'hour': 'H',
        'day': 'D',
        'week': 'W',
        'month': 'M'
    }
    
    # Group by the specified period
    grouped = df.groupby(pd.Grouper(key='Timestamp', freq=period_mapping[period])).agg({
        'Battery_Charging': 'sum',
        'Battery_Discharging': 'sum',
        'Battery_Level': 'mean',  # Average battery level for the period
        'Battery_Revenue': 'sum',
        'Battery_Cost': 'sum',
        'Fixed_Cost': 'sum'
    }).reset_index()
    
    return grouped

def create_battery_operation_plot(df, station_name):
    """Create an improved time-aware visualization of battery operations"""
    # Add time period selector
    time_period = st.selectbox(
        "Select Time Period",
        ["Hourly", "Daily", "Weekly", "Monthly"],
        index=1  # Default to daily view
    )
    
    # Map selection to aggregation period
    period_map = {
        "Hourly": "hour",
        "Daily": "day",
        "Weekly": "week",
        "Monthly": "month"
    }
    
    # Aggregate data based on selected period
    agg_df = aggregate_battery_data(df, period_map[time_period])
    
    # Create figure
    fig = go.Figure()
    
    # Convert power values for visualization
    charging_power = agg_df['Battery_Charging']
    discharging_power = -agg_df['Battery_Discharging']  # Negative to show below axis
    
    # Add charging trace (positive values)
    fig.add_trace(go.Bar(
        x=agg_df['Timestamp'],
        y=charging_power,
        name='Charging',
        marker_color='green',
        opacity=0.7,
        hovertemplate=(
            "%{x}<br>" +
            "Charging: %{y:.1f} MWh<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add discharging trace (negative values)
    fig.add_trace(go.Bar(
        x=agg_df['Timestamp'],
        y=discharging_power,
        name='Discharging',
        marker_color='red',
        opacity=0.7,
        hovertemplate=(
            "%{x}<br>" +
            "Discharging: %{y:.1f} MWh<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add battery level trace
    fig.add_trace(go.Scatter(
        x=agg_df['Timestamp'],
        y=agg_df['Battery_Level'],
        name='Avg Battery Level',
        line=dict(color='lightblue', width=2),
        yaxis='y2',
        hovertemplate=(
            "%{x}<br>" +
            "Avg Level: %{y:.1f} MWh<br>" +
            "<extra></extra>"
        )
    ))
    
    # Calculate axis ranges with padding
    y_max = max(charging_power.max(), abs(discharging_power.min())) * 1.1
    y_min = -y_max  # Make y-axis symmetric
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{station_name} - Battery Operations ({time_period} View)",
            x=0.5,
            font=dict(size=20)
        ),
        barmode='relative',
        yaxis=dict(
            title=f"Energy ({time_period.rstrip('ly')}ly Total MWh)",
            side="left",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='grey',
            gridcolor='rgba(128,128,128,0.2)',
            range=[y_min, y_max]
        ),
        yaxis2=dict(
            title="Average Battery Level (MWh)",
            side="right",
            overlaying="y",
            showgrid=False,
            range=[0, agg_df['Battery_Level'].max() * 1.1]
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='grey',
            rangeslider=dict(visible=True)  # Add range slider for time navigation
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=50, b=50)
    )

    # Add date range selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

def add_summary_metrics(df, period):
    """Add summary metrics for the selected time period"""
    col1, col2, col3 = st.columns(3)

    with col1:
        total_charged = df['Battery_Charging'].sum()
        st.metric(
            f"Total Energy Charged ({period})",
            f"{total_charged:,.1f} MWh"
        )

    with col2:
        total_discharged = df['Battery_Discharging'].sum()
        st.metric(
            f"Total Energy Discharged ({period})",
            f"{total_discharged:,.1f} MWh"
        )

    with col3:
        avg_level = df['Battery_Level'].mean()
        st.metric(
            "Average Battery Level",
            f"{avg_level:,.1f} MWh"
        )

# Update the relevant section in your main() function:
def update_battery_operations_tab(df_with_battery, station_name):
    """Update the battery operations tab with time-aware visualizations"""
    st.plotly_chart(create_battery_operation_plot(df_with_battery, station_name), use_container_width=True)

    # Get current time period from session state or default to 'Daily'
    current_period = st.session_state.get('time_period', 'Daily')

    # Add summary metrics
    add_summary_metrics(df_with_battery, current_period)

def main():
    st.title("ChargeMax")

    # Sidebar controls
    st.sidebar.header("Controls")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        df = load_and_preprocess_data(uploaded_file)

        # Station name input
        station_name = st.text_input("Power Station Name", "Valentino")

        # Battery configuration
        st.header("Battery Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            battery = st.selectbox(
                "Lithium Ion Battery technology",
                ["LMO/Graphite", "LFP/Graphite", "LCO/Graphite", "LMO/LTO", "NMC/Graphite", "NCA/Graphite", "Custom"],
                index=0
            )

        with col2:
            battery_capacity_MW = st.number_input(
                "Battery capacity (MW)",
                value=4000,
                min_value=1000,
                max_value=10000,
                step=100
            )

        with col3:
            battery_capacity_h = st.selectbox(
                "Battery capacity (h)",
                [2, 4, 6, 8],
                index=1
            )

        fixed_cost = {2: 30, 4: 26, 6: 23, 8: 20}
        st.code('Fixed cost per MWh of battery capacity: $' + str(fixed_cost[battery_capacity_h]))

        # Calculate battery capacity
        battery_capacity = battery_capacity_h * battery_capacity_MW

        # Run battery optimization
        df_with_battery = optimize_battery_operations(df, battery_capacity_MW, battery_capacity_h, fixed_cost[battery_capacity_h])

        # Display results
        tab1, tab2, tab3 = st.tabs(["📊 Battery Operations", "💰 Revenue Analysis", "📈 Performance Metrics"])

        with tab1:
            # Battery operation visualization
            st.plotly_chart(create_battery_operation_plot(df_with_battery, station_name), use_container_width=True)

            # Daily charging source breakdown
            charging_source_stats = df_with_battery.groupby('Charging_Source')['Battery_Charging'].sum()
            fig = px.pie(
                values=charging_source_stats,
                names=charging_source_stats.index,
                title="Charging Source Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Revenue metrics
            col1, col2 = st.columns(2)

            with col1:
                total_revenue = df_with_battery['Battery_Revenue'].sum()
                st.metric(
                    "Total Battery Revenue ($)",
                    f"{total_revenue:,.2f}"
                )
                total_charging_cost = df_with_battery['Battery_Cost'].sum()
                st.metric(
                    "Total Charging Cost ($)",
                    f"{total_charging_cost:,.2f}"
                )
                total_fixed_cost = df_with_battery['Fixed_Cost'].sum()
                st.metric(
                    "Total Fixed Cost ($)",
                    f"{total_fixed_cost:,.2f}"
                )

            with col2:
                revenue_per_mwh = total_revenue / battery_capacity
                st.metric(
                    "Revenue per MWh of Battery Capacity ($/MWh)",
                    f"{revenue_per_mwh:,.2f}"
                )

                dr = st.selectbox('Discount rate %', [5, 6, 7, 8, 9, 10])

                lifetime = st.selectbox('Project lifetime (yrs)', [5, 10, 15, 20, 25, 30])

                deg_rate = st.selectbox('Degradation rate %', [0.1, 0.2, 0.3, 0.4, 0.5])

                # Calculate NPV
                annualized_revenue = list((df_with_battery.groupby('Year')['Battery_Revenue'].sum().reset_index())['Battery_Revenue'])
                annualized_cost = df_with_battery.groupby('Year')['Battery_Cost'].sum().reset_index()
                annualized_fixed_cost = df_with_battery.groupby('Year')['Fixed_Cost'].sum().reset_index()
                total_annual_cost = list(annualized_cost['Battery_Cost'] + annualized_fixed_cost['Fixed_Cost'])
                print('revenue', annualized_revenue)
                print('cost', total_annual_cost)
                cashflow = [annualized_revenue[i] - total_annual_cost[i] for i in range(len(annualized_revenue))]
                print('cashflow', cashflow)
                while len(cashflow) < lifetime:
                    cashflow.append(cashflow[-1] * (1- deg_rate/100) )
                print('augmented cashflow', cashflow)
                npv = npf.npv(dr, cashflow)
                st.metric(
                    "Net Present Value ($)",
                    f"{npv:,.2f}"
                )


            # Daily revenue plot
            # Grouping and summing data
            daily_revenue = df_with_battery.groupby('Date')['Battery_Revenue'].sum().reset_index()
            daily_cost = df_with_battery.groupby('Date')['Battery_Cost'].sum().reset_index()
            fixed_cost = df_with_battery.groupby('Date')['Fixed_Cost'].sum().reset_index()

            # Create a figure
            fig = go.Figure()

            # Add each trace with distinct colors
            fig.add_trace(go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['Battery_Revenue'],
                mode='lines',
                name='Battery Revenue',
                line=dict(color='blue')  # Specify color
            ))
            fig.add_trace(go.Scatter(
                x=daily_cost['Date'],
                y=daily_cost['Battery_Cost'],
                mode='lines',
                name='Battery Cost',
                line=dict(color='red')  # Specify color
            ))
            fig.add_trace(go.Scatter(
                x=fixed_cost['Date'],
                y=fixed_cost['Fixed_Cost'],
                mode='lines',
                name='Fixed Cost',
                line=dict(color='green')  # Specify color
            ))

            # Add title and layout adjustments
            fig.update_layout(
                title="Revenue and Cost",
                xaxis_title="Date",
                yaxis_title="Amount",
                legend_title="Legend",
            )

            # Render the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            # Performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                cycles = df_with_battery['Battery_Charging'].sum() / battery_capacity
                st.metric(
                    "Total Battery Cycles",
                    f"{cycles:.1f}"
                )

            with col2:
                utilization = (df_with_battery['Battery_Level']/battery_capacity).mean() * 100
                st.metric(
                    "Battery Utilization (%)",
                    f"{utilization:.1f}"
                )

            with col3:
                curtailment_captured = (df_with_battery[df_with_battery['Charging_Source'] == 'Curtailment']['Battery_Charging'].sum() / 
                                      df_with_battery['Curtailment (MWh)'].sum() * 100)
                st.metric(
                    "Curtailment Captured (%)",
                    f"{curtailment_captured:.1f}"
                )

            # Battery level distribution
            fig = px.histogram(
                df_with_battery,
                x='Battery_Level',
                title="Battery Level Distribution",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="ChargeMax",
        page_icon="🔋",
        layout="wide"
    )
    main()