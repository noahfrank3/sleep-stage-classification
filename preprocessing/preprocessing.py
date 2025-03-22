import pandas as pd

from pathlib import Path
from create_cassette_data import create_cassette_data
from create_telemetry_data import create_telemetry_data

# Create cassette and telemetry data
cassette_data = create_cassette_data()
telemetry_data = create_telemetry_data()

# Save data
path = Path('../data')
X.to_csv(path / 'X.csv', index=False)
y.to_csv(path / 'y.csv', index=False)
y_extras.to_csv(path / 'y_extras.csv', index=False)
