import argparse
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objs as go
import sys
import os
import numpy as np

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot GTSAM results CSV data versus time in 5x3 interactive subplots.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    args = parser.parse_args()

    csv_path = args.csv_path

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Read CSV data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Check for required columns
    expected_columns = ['time(s)', 'x(m)', 'y(m)', 'z(m)', 'Roll(deg)', 'Pitch(deg)', 'Yaw(deg)',
                       'vx(mps)', 'vy(mps)', 'vz(mps)',
                       'bgx(deg/s)', 'bgy(deg/s)', 'bgz(deg/s)',
                       'bax(m/s^2)', 'bay(m/s^2)', 'baz(m/s^2)']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Missing column in CSV: {col}")
            sys.exit(1)

    time = df['time(s)']

    # Subplot configuration: 5 rows x 3 columns, shared x-axis
    subplot_titles = [
        "Position X (m)", "Position Y (m)", "Position Z (m)",
        "Velocity X (m/s)", "Velocity Y (m/s)", "Velocity Z (m/s)",
        "Accel Bias X (m/s²)", "Accel Bias Y (m/s²)", "Accel Bias Z (m/s²)",
        "Roll (deg)", "Pitch (deg)", "Yaw (deg)",
        "Gyro Bias X (deg/s)", "Gyro Bias Y (deg/s)", "Gyro Bias Z (deg/s)"
    ]
    
    fig = sp.make_subplots(
        rows=5, cols=3,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        subplot_titles=subplot_titles
    )

    # Data series: (column name, row, col)
    data_series = [
        # Position row
        ('x(m)', 1, 1),
        ('y(m)', 1, 2),
        ('z(m)', 1, 3),
        # Velocity row
        ('vx(mps)', 2, 1),
        ('vy(mps)', 2, 2),
        ('vz(mps)', 2, 3),
        # Accel bias row
        ('bax(m/s^2)', 3, 1),
        ('bay(m/s^2)', 3, 2),
        ('baz(m/s^2)', 3, 3),
        # Euler angles row
        ('Roll(deg)', 4, 1),
        ('Pitch(deg)', 4, 2),
        ('Yaw(deg)', 4, 3),
        # Gyro bias row
        ('bgx(deg/s)', 5, 1),
        ('bgy(deg/s)', 5, 2),
        ('bgz(deg/s)', 5, 3)
    ]

    for idx, (col_name, row, col) in enumerate(data_series):
        trace = go.Scatter(
            x=time,
            y=df[col_name],
            mode='lines',
            name=col_name,
            line=dict(width=1.5)
        )
        fig.add_trace(trace, row=row, col=col)

    # Set x-axis label only on bottom row
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="Time (s)", row=5, col=col)

    # Layout configuration with synchronized time axes
    fig.update_layout(
        title_text="GTSAM Results vs Time",
        title_x=0.5,
        height=1200,
        width=1400,
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        # Synchronize all x-axes
        xaxis=dict(matches='x'),
        xaxis2=dict(matches='x'),
        xaxis3=dict(matches='x'),
        xaxis4=dict(matches='x'),
        xaxis5=dict(matches='x'),
        xaxis6=dict(matches='x'),
        xaxis7=dict(matches='x'),
        xaxis8=dict(matches='x'),
        xaxis9=dict(matches='x'),
        xaxis10=dict(matches='x'),
        xaxis11=dict(matches='x'),
        xaxis12=dict(matches='x'),
        xaxis13=dict(matches='x'),
        xaxis14=dict(matches='x'),
        xaxis15=dict(matches='x')
    )
    
    # Set y-axis autorange for all subplots
    fig.update_yaxes(autorange=True)
    
    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.show()

if __name__ == "__main__":
    main()
