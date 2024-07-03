#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.initializers import glorot_uniform
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
# from Auto_encoders import AutoEncoder
# from preprocessing import DataPreprocessor

class CorrelationAnalysis:
    def __init__(self, autoencoder_input_data, autoencoder_weights,enso_csv):
        self.autoencoder_input_data = autoencoder_input_data
        self.autoencoder_weights = autoencoder_weights
        self.enso_csv = enso_csv
        self.pred = None

    def _analyze_correlation(self):
        threshold_values = self.Tweights(self.autoencoder_weights)
        self.pred = self.potential_pred(threshold_values, self.autoencoder_input_data, self.autoencoder_weights)
    
    def correlation_methods(self, correlation_method):
        if correlation_method == 'mutual':
            top_preds, sorted_corr_data = self.mut_corr(self.pred, self.enso_csv)
        else:
            top_preds, sorted_corr_data = self.corr(self.pred, self.enso_csv, correlation_method)

        return top_preds, sorted_corr_data
    
    def Tweights(self,weight):
        threshold_values = []
        for threshold_multiplier in np.arange(2, 1, -0.1):
            num = []
            for i in range(weight.shape[1]):
                weight_mean = np.mean(weight[:, i])
                weight_std = np.std(weight[:, i])
                threshold_upper = weight_mean + threshold_multiplier * weight_std
                threshold_lower = weight_mean - threshold_multiplier * weight_std
                nodes_with_weight_above_upper_threshold = np.sum(weight[:, i] > threshold_upper)
                nodes_with_weight_below_lower_threshold = np.sum(weight[:, i] < threshold_lower)
                ten_percent_nodes = int(0.1 * weight.shape[0])

                if (nodes_with_weight_above_upper_threshold + nodes_with_weight_below_lower_threshold) > ten_percent_nodes:
                    num.append(nodes_with_weight_above_upper_threshold + nodes_with_weight_below_lower_threshold)
                    #print(i, nodes_with_weight_above_upper_threshold, threshold_upper, nodes_with_weight_below_lower_threshold, threshold_lower)

                    if len(num) == weight.shape[1]:
                        threshold_values.append(threshold_multiplier)
                        break
            if len(num) == weight.shape[1]:
                break

        return threshold_values
    
    def potential_pred(self,threshold_values,input_data,weights):
        pred = np.zeros((65,input_data.shape[0]))
        for i in range(weights.shape[1]):
            weight_mean = np.mean(weights[:,i])
            weight_std = np.std(weights[:,i])
            threshold_upper = weight_mean + threshold_values[0] * weight_std
            threshold_lower = weight_mean - threshold_values[0] * weight_std
            nodes_with_weight_above_upper_threshold = np.sum(weights[:, i] > threshold_upper)
            nodes_with_weight_below_lower_threshold = np.sum(weights[:, i] < threshold_lower)
            ten_percent_nodes = int(0.1 * weights.shape[0])
            if (nodes_with_weight_above_upper_threshold + nodes_with_weight_below_lower_threshold) > ten_percent_nodes:
                for h in range(input_data.shape[0]): 
                    pred_i = 0
                    for j in range(weights.shape[0]):
                        weight_value = weights[j,i]
                        if weight_value > threshold_upper or weight_value < threshold_lower:
                            pp = np.sum(weight_value*input_data[h,j])
                            pred_i += pp
                            pred[i,h] = pred_i
        return pred
    
    def corr(self,data,csv,correlation_method):
        global top_correlation
        top_correlation = []
        correlation_data = []
        top_pred = pd.DataFrame()

        for i in range(data.shape[0]):
            pred_pres = data[i,:].reshape(int(data.shape[1]/12),12)
            years = pd.date_range(start='1958', end='2023', freq='YS').year
            months = pd.date_range(start='1975-01', periods=12, freq='MS').strftime('%B')
            df_pres = pd.DataFrame(pred_pres, index=years, columns=months)
            df_pres = df_pres.reset_index()

            df = pd.read_csv(csv)
            df['jjas_Avg'] = df[['Jun', 'Jul','Aug','Sep']].mean(axis=1)
            df_pres['enso_avg'] = df['jjas_Avg'].copy()
            #month_column_index = df_pres.columns.get_loc(month)
            new_row = [0] * (2)
            new_row.extend(df_pres.iloc[0, 2 + 1:-1].tolist())
            new_row = [1957] + new_row + [df_pres['enso_avg'][0]]

            df_pres = pd.concat([pd.DataFrame([new_row], columns=df_pres.columns), df_pres], ignore_index=True)
            df_pres= df_pres.drop('index',axis=1)
            df_6 = df_pres.iloc[1:, :2]
            df_6 = df_6.reset_index()
            df_12 = df_pres.iloc[0:-1, 2:12]
            df_12 = df_12.reset_index()
            df_13 = df_pres.iloc[1:, [12]]
            df_13 = df_13.reset_index()
            df_last = pd.concat([df_6, df_12, df_13], axis=1)
            df_last = df_last.drop('index',axis=1)

            correlation = df_last.corr(method=correlation_method)
            second_max_value = correlation['enso_avg'].sort_values(ascending=False)[1]
            negative_minimum = correlation['enso_avg'].sort_values(ascending=False)[-1]
            second_max_index = correlation['enso_avg'].sort_values(ascending=False).index[1]
            negative_min_index = correlation['enso_avg'].sort_values(ascending=False).index[-1]

            if abs(second_max_value) > abs(negative_minimum):
                highest_correlation = second_max_value
                highest_correlation_index = second_max_index

            else:
                highest_correlation = negative_minimum
                highest_correlation_index = negative_min_index

            column_name = highest_correlation_index
            column_index = df_last.columns.get_loc(highest_correlation_index)
            correlation_data.append((i, column_index, abs(highest_correlation)))
            suffix_index = 1

            while column_name in top_pred.columns:
                suffix_index += 1
                column_name = f"{highest_correlation_index}_{suffix_index}"

            top_pred[column_name] = df_last[highest_correlation_index]
            top_correlation.append((i, highest_correlation,column_name , abs(highest_correlation)))
            #print(i,highest_correlation,column_name,abs(highest_correlation))
            result_df = pd.DataFrame(top_correlation, columns=['Iteration', 'Correlation', 'Column_Name', 'Absolute_Correlation'])
            result_df = result_df.sort_values(by='Absolute_Correlation', ascending=False)
            unique_values_list_df = result_df['Column_Name'].unique().tolist()
            top_preds = top_pred[unique_values_list_df]
            sorted_correlation_data = sorted(correlation_data, key=lambda x: x[2], reverse=True)
        return top_preds,sorted_correlation_data
    
    def mut_corr(self,data,csv):
        global top_correlation
        top_correlation = []
        correlation_data = []
        top_pred = pd.DataFrame()
        for i in range(data.shape[0]):
            pred_pres = data[i,:].reshape(int(data.shape[1]/12),12)
            years = pd.date_range(start='1958', end='2023', freq='YS').year
            months = pd.date_range(start='1975-01', periods=12, freq='MS').strftime('%B')
            df_pres = pd.DataFrame(pred_pres, index=years, columns=months)
            df_pres = df_pres.reset_index()
            df = pd.read_csv(csv)
            
            df['jjas_Avg'] = df[['Jun', 'Jul','Aug','Sep']].mean(axis=1)
            df_pres['enso_avg'] = df['jjas_Avg'].copy()
            #month_column_index = df_pres.columns.get_loc(month)
            new_row = [0] * (5)
            new_row.extend(df_pres.iloc[0, 5 + 1:-1].tolist())
            new_row = [1957] + new_row + [df_pres['enso_avg'][0]]

            df_pres = pd.concat([pd.DataFrame([new_row], columns=df_pres.columns), df_pres], ignore_index=True)
            df_pres= df_pres.drop('index',axis=1)
            df_6 = df_pres.iloc[1:, :5]
            df_6 = df_6.reset_index()
            df_12 = df_pres.iloc[0:-1, 5:12]
            df_12 = df_12.reset_index()
            df_13 = df_pres.iloc[1:, [12]]
            df_13 = df_13.reset_index()
            df_last = pd.concat([df_6, df_12, df_13], axis=1)
            df_last = df_last.drop('index',axis=1)

            # Calculate mutual information
            features = df_last.drop('enso_avg', axis=1)
            target = df_last['enso_avg']
            mutual_info = mutual_info_regression(features, target)

            highest_mutual_info_index = np.argmax(mutual_info)
            highest_mutual_info_value = mutual_info[highest_mutual_info_index]
            highest_mutual_info_column = df_last.columns[highest_mutual_info_index]

            column_name = highest_mutual_info_column
            column_index = highest_mutual_info_index
            correlation_data.append((i, column_index, abs(highest_mutual_info_value)))
            suffix_index = 1
            while column_name in top_pred.columns:
                suffix_index += 1
                column_name = f"{highest_mutual_info_column}_{suffix_index}"

            top_pred[column_name] = df_last[highest_mutual_info_column]
            top_correlation.append((i, highest_mutual_info_value, column_name, abs(highest_mutual_info_value)))
            #print(i, highest_mutual_info_value, column_name, abs(highest_mutual_info_value))
            result_df = pd.DataFrame(top_correlation, columns=['Iteration', 'Correlation', 'Column_Name', 'Absolute_Correlation'])
            result_df = result_df.sort_values(by='Absolute_Correlation', ascending=False)
            unique_values_list_df = result_df['Column_Name'].unique().tolist()
            top_preds = top_pred[unique_values_list_df]
            sorted_correlation_data = sorted(correlation_data, key=lambda x: x[2], reverse=True)

        return top_preds,sorted_correlation_data
    
    def run_correlation_analysis(self):
        top_predictions = {}
        sorted_correlation_data = {}

        self._analyze_correlation()  # Call the _analyze_correlation method

        for method in ['pearson', 'kendall', 'spearman', 'mutual']:
            top_preds, sorted_corr_data = self.correlation_methods(method)
            top_predictions[f'top_pred_{method}'] = top_preds
            sorted_correlation_data[f'sorted_correlation_{method}'] = sorted_corr_data

        return top_predictions, sorted_correlation_data






