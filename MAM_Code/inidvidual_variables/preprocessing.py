


import netCDF4 as nc
from netCDF4 import num2date,date2index
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings("ignore")




import numpy as np
import netCDF4 as nc
from netCDF4 import num2date
import pandas as pd

class DataPreprocessor:
    def __init__(self, nc_file):
        self.nc_file = nc_file

    def preprocessing(self):
        data = nc.Dataset(self.nc_file)
        variable_names = list(data.variables.keys())
        latitude = data.variables[variable_names[2]][:]
        longitude = data.variables[variable_names[1]][:]
        time_var = data.variables['time']
        dates = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
        variable = data.variables[variable_names[3]][:, :, :]
        data_ = np.array(variable)
        data_[data_ == -9.96921e+36] = 0
        data_reshaped = np.reshape(data_, (len(data_) // 12, 12, data_.shape[1], data_.shape[2]))
        print("Start date:", dates[0])
        print("End date:", dates[-1])
        return data_reshaped, latitude, longitude

    def lon_pp(self, longitude):
        lon_indices = []
        for start_value in np.arange(0, 360, 20):
            end_value = start_value + 20
            range_indices = np.where((longitude >= start_value) & (longitude < end_value))[0]
            if len(range_indices) > 0:
                min_index = range_indices[np.argmin(longitude[range_indices])]
                max_index = range_indices[np.argmax(longitude[range_indices])]
                lon_indices.append(min_index)
                lon_indices.append(max_index)
        return lon_indices

    def lat_pp(self, latitude):
        lat_indices = []
        for start_value in np.arange(90, -90, -10):
            end_value = start_value - 10
            range_indices = np.where((latitude <= start_value) & (latitude > end_value))[0]
            if len(range_indices) > 0:
                min_index = range_indices[np.argmax(latitude[range_indices])]
                max_index = range_indices[np.argmin(latitude[range_indices])]
                lat_indices.append(min_index)
                lat_indices.append(max_index)
        return lat_indices

    def coarse_gridding(self, data, lat_indices, lon_indices):
        coarsed_data = np.zeros((data.shape[0], data.shape[1], len(lat_indices) // 2, len(lon_indices) // 2))
        for lat_idx in range(0, len(lat_indices), 2):
            lat_range_start = lat_indices[lat_idx]
            lat_range_end = lat_indices[lat_idx + 1]
            for lon_idx in range(0, len(lon_indices), 2):
                lon_range_start = lon_indices[lon_idx]
                lon_range_end = lon_indices[lon_idx + 1]
                subset = data[:, :, lat_range_start:lat_range_end, lon_range_start:lon_range_end]
                averaged_value = np.mean(subset, axis=(2, 3))
                coarsed_data[:, :, lat_idx // 2, lon_idx // 2] = averaged_value
        final = np.reshape(coarsed_data, (len(coarsed_data), 12, coarsed_data.shape[2]*coarsed_data.shape[3]))
        return final

    def calculate_monthly_anomalies(self, data):
        anomalies = np.zeros(data.shape)
        for i in range(data.shape[2]):
            for j in range(data.shape[1]):
                month = data[:, j, i]
                monthly_mean = np.mean(month)
                anomalies[:, j, i] = month - monthly_mean
        return anomalies

    def min_max_normalize(self, data):
        normalized_data = np.zeros(data.shape)
        for i in range(data.shape[2]):
            for j in range(data.shape[1]):
                month = data[:, j, i]
                min_value = np.min(month)
                max_value = np.max(month)
                normalized_month = (month - min_value) / (max_value - min_value)
                normalized_data[:, j, i] = normalized_month
        return normalized_data

    def execute_pipeline(self):
        data_reshaped, latitude, longitude = self.preprocessing()
        lon_indices = self.lon_pp(longitude)
        lat_indices = self.lat_pp(latitude)
        coarsed_data = self.coarse_gridding(data_reshaped, lat_indices, lon_indices)
        anomalies = self.calculate_monthly_anomalies(coarsed_data)
        normalized_data = self.min_max_normalize(anomalies)
        return normalized_data







