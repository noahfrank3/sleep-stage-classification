from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(ABC):
    db_path_mappings = {
            'zip': Path('..', 'sleep-edf-database-expanded-1.0.0'),
            'wget': Path('..', 'physionet.org', 'files', 'sleep-edfx', '1.0.0')
    }
    study_name = None
    column_mappings = None
    sex_mappings = None

    def __init__(self, db_path):
        self.db_path = db_path
        self.data_path = self.db_path / ('sleep-' + self.__class__.study_name)
        self.column_mappings = self.__class__.column_mappings
        self.sex_mappings = self.__class__.sex_mappings

    def get_data(self):
        self.load_data()
        self.rename_columns()
        self.reorganize_data()
        self.fix_sex_labels()
        self.reorder_columns()
        
        return self.data

    @abstractmethod
    def load_data(self):
        pass

    def rename_columns(self):
        self.data = self.data.rename(columns=self.column_mappings)

    @abstractmethod
    def reorganize_data(self):
        pass

    def fix_sex_labels(self):
        self.data['sex'] = self.data['sex'].astype(str)

        for key, sex in self.sex_mappings.items():
            self.data.loc[self.data['sex'] == key, 'sex'] = sex

    def reorder_columns(self):
        del self.data['subject']

        self.data = self.data[[
            'study',
            'night',
            'age',
            'sex',
            'lights_off'
        ]]

class CassettePreprocessor(Preprocessor):
    study_name = 'cassette'
    column_mappings = {
            'k': 'subject',
            'sex (F=1)': 'sex',
            'LightsOff': 'lights_off'
    }
    sex_mappings = {
            '1': 'F',
            '2': 'M'
    }

    def load_data(self):
        self.data = pd.read_excel(self.db_path / 'SC-subjects.xls')

    def reorganize_data(self):
        self.data['study'] = 'cassette'

class TelemetryPreprocessor(Preprocessor):
    study_name = 'telemtry'
    column_mappings = {
            'Nr': 'subject',
            'Age': 'age',
            'M1/F2': 'sex',
            'night nr': 'placebo_night',
            'lights off': 'placebo_lights_off',
            'night nr.1': 'temazepam_night',
            'lights off.1': 'temazepam_lights_off'
    }
    sex_mappings = {
            '1': 'M',
            '2': 'F'
    }

    def load_data(self):
        self.data = pd.read_excel(self.db_path / 'ST-subjects.xls',
                                  skiprows=1)

    def reorganize_data(self):
        # Remove other dataset's features
        placebo_data = self.data.drop(
                ['temazepam_night', 'temazepam_lights_off'], axis=1)
        temazepam_data = self.data.drop(
                ['placebo_night', 'placebo_lights_off'], axis=1)

        # Create common column naming
        placebo_data = placebo_data.rename(columns={
            'placebo_night': 'night',
            'placebo_lights_off': 'lights_off'
        })
        temazepam_data = temazepam_data.rename(columns={
            'temazepam_night': 'night',
            'temazepam_lights_off': 'lights_off'
        })

        # Define study
        placebo_data['study'] = 'placebo'
        temazepam_data['study'] = 'temazepam'
        
        # Recombine placebo and temazepam data
        self.data = pd.concat([placebo_data, temazepam_data],
                              ignore_index=True)
        self.data = self.data.sort_values(by=['subject', 'night'],
                                          ignore_index=True)

def circularize_time(data) :
    time_encoder = CircularEncoder()
    
    # Transform datetime to float
    data['lights_off'] = data['lights_off'].apply(
            lambda t: (t.hour + t.minute/60)/24)
    
    # Get cos and sin components
    data['lights_off_cos'], data['lights_off_sin'] = \
                time_encoder.transform(data['lights_off'])
    
    # Remove old entry
    del data['lights_off']

    return data

class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period=1):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        theta = 2*np.pi*X/self.period
        return np.cos(theta), np.sin(theta)

    def inverse_transform(self, X):
        X = np.asarray(X)
        theta = np.arctan2(X[:, 1], X[:, 0])
        return theta*self.period/(2*np.pi)
