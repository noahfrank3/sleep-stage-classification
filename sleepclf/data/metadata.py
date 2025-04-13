from abc import ABC, abstractmethod
import logging
import os

import pandas as pd

from sleepclf.config import EDF_DATA_PATH, DATA_PATH

METADATA_PATH = os.path.join(DATA_PATH, 'metadata.csv')

class Preprocessor(ABC):
    study_name = None
    column_mappings = None
    sex_mappings = None

    def __init__(self):
        self.column_mappings = self.__class__.column_mappings
        self.sex_mappings = self.__class__.sex_mappings

    @abstractmethod
    def load_data(self):
        pass

    def rename_columns(self):
        self.data = self.data.rename(columns=self.column_mappings)

    @abstractmethod
    def reorganize_data(self):
        pass

    def convert_nights(self):
        self.data['night'] = self.data['night'].astype(int)
        self.data['night'] = self.data['night'] - 1

    def convert_ages(self):
        self.data['age'] = self.data['age'].astype(int)

    def fix_sex_labels(self):
        self.data['sex'] = self.data['sex'].astype(str)

        for key, sex in self.sex_mappings.items():
            self.data.loc[self.data['sex'] == key, 'sex'] = sex

    def convert_datetimes(self):
        self.data['lights_off'] = self.data['lights_off'].apply(
                lambda time: (time.hour + time.minute/60)/24)

    def reorder_columns(self):
        del self.data['subject']

        self.data = self.data[[
            'study',
            'night',
            'age',
            'sex',
            'lights_off'
        ]]

    def get_data(self):
        self.load_data()
        self.rename_columns()
        self.reorganize_data()
        self.convert_nights()
        self.convert_ages()
        self.fix_sex_labels()
        self.convert_datetimes()
        self.reorder_columns()
        
        return self.data

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
        self.data = pd.read_excel(os.path.join(
            EDF_DATA_PATH,
            'SC-subjects.xls'
        ))

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
        self.data = pd.read_excel(
                os.path.join(EDF_DATA_PATH, 'ST-subjects.xls'),
                skiprows=1
        )

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

def create_metadata():
    # Create metadata preprocessors
    cassette_preprocessor = CassettePreprocessor()
    telemtry_preprocessor = TelemetryPreprocessor()

    # Preprocess cassette and telemtry metadata
    cassette_data = cassette_preprocessor.get_data()
    telemetry_data = telemtry_preprocessor.get_data()

    # Combine cassette and telemtry metadata
    data = pd.concat([cassette_data, telemetry_data], ignore_index=True)
    
    data.to_csv(METADATA_PATH, index=False)
    return data

def load_metadata():
    return pd.read_csv(METADATA_PATH)
