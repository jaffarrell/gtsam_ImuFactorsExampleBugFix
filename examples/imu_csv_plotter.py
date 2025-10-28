import argparse
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objs as go
import sys
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot CSV data versus time in 3x2 interactive subplots.")
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
    expected_columns = ['time_since_start', 'delta_vx', 'delta_vy', 'delta_vz',
                        'delta_thx', 'delta_thy', 'delta_thz']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Missing column in CSV: {col}")
            sys.exit(1)

    time = df['time_since_start']

    # Subplot configuration: 3 rows x 2 columns, shared x-axis
    subplot_titles = [
        "ΔVx, mps", "Δθx, rad",
        "ΔVy, mps", "Δθy, rad",
        "ΔVz, mps", "Δθz, rad"
    ]
    fig = sp.make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.2  # Increase column spacing
    )

    # Data series: (column name, row, col)
    data_series = [
        ('delta_vx', 1, 1),
        ('delta_thx', 1, 2),
        ('delta_vy', 2, 1),
        ('delta_thy', 2, 2),
        ('delta_vz', 3, 1),
        ('delta_thz', 3, 2)
    ]

    for idx, (col_name, row, col) in enumerate(data_series):
        trace = go.Scatter(
            x=time,
            y=df[col_name],
            mode='lines'
        )
        fig.add_trace(trace, row=row, col=col)

        # Set y-axis label from subplot_titles
        fig.update_yaxes(title_text=subplot_titles[idx], row=row, col=col)

    # Set x-axis label only on bottom row
    for col in [1, 2]:
        fig.update_xaxes(title_text="Time (s)", row=3, col=col)

    # Layout configuration
    fig.update_layout(
        title_text="IMU Measurement Data vs Time",
        title_x = 0.5,
        height=900,
        width=1000,
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
        xaxis_rangeslider_visible=False,  
        xaxis2=dict(matches='x'),
        xaxis3=dict(matches='x'),
        xaxis4=dict(matches='x'),
        xaxis5=dict(matches='x'),
        xaxis6=dict(matches='x'),
    )
    fig.update_yaxes(autorange=True)
    fig.show()

if __name__ == "__main__":
    main()