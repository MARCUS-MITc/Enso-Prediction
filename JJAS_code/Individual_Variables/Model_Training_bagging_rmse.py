

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os



class ENSOAnalysis_br:
    def __init__(self, correlations, enso_index_file):
        self.correlations = correlations
        self.enso_index_file = enso_index_file
        self.best_models = {} 
        self.scalers = {} 
        self.y_preds = {} 
        self.y_tests = {} 

    
    
    def processing(self,csv,correaltion,features,month):
        df = pd.read_csv(csv)
        df['jjas_Avg'] = df[['Jun', 'Jul','Aug','Sep']].mean(axis=1)
        scaler = StandardScaler()
        x_train = correaltion.iloc[:50, :features]
        x_test = correaltion.iloc[50:, :features]
        y_train = df.iloc[:50][month]
        y_test = df.iloc[50:][month]
        X_train = scaler.fit_transform(x_train)
        X_test = scaler.fit_transform(x_test)
        self.scalers[f"{month}"] = scaler
        self.y_tests[f"{month}"] = y_test
        return X_train,X_test,y_train,y_test
    
    

    def tune_random_forest_regressor(self, X_train, y_train):
        
        param_grid = {
            'n_estimators': range(10, 100),
            'max_samples': np.arange(0.1, 1.01, 0.1),
            
         
        }
        
        random_forest_reg = BaggingRegressor(oob_score=True, random_state=42)
        grid_search = RandomizedSearchCV(random_forest_reg, param_grid,n_iter=300,random_state=42, cv=3, scoring='r2', n_jobs=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_model, best_params, best_score

    
    def model_bag_reg(self,best_model, X_test, y_test):
        score = best_model.score(X_test, y_test)  
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse, y_pred
    
    def plotting(self,y_test, y_pred, month, corr_coef, features,corr_name):
        actual_values = y_test
        predicted_values = y_pred
        years = range(2008, 2024)
        plt.figure(figsize=(8, 6))
        plt.scatter(actual_values, predicted_values, s=30, c='black')
        for actual, predicted, year in zip(actual_values, predicted_values, years):
            plt.annotate(f"{year}", (actual, predicted), xytext=(-1, -15), textcoords='offset points',fontsize=8)
        plt.plot([-2, 2], [-2, 2], color='red', linestyle='--', label=f'RMSE: {corr_coef:.2f}')
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Air_temp {month} with {corr_name} correaltion and {features} features', fontsize=14)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.grid(True)
        plt.legend()
        folder_name = "Bagging_RMSE"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the scatter plot
        plot_filename = f"{folder_name}/{month}_scatter_plot.png"
        plt.savefig(plot_filename)
        plt.show()
    
    def plotting_y_pred_vs_y_test(self, y_test, y_pred, month):
        actual_values = y_test
        predicted_values = y_pred
        years = list(range(2008, 2024))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Bar plot for y_test
        ax.bar(years, actual_values, label='Actual Values')
        
        # Line plot with scatter for y_pred
        ax.plot(years, predicted_values, marker='o', color='red', linestyle='-', label='Predicted Values')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Values')
        ax.set_title(f'Air_Temp {month} Actual vs Predicted')
        ax.legend()
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        folder_name = "Bagging_RMSE"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the scatter plot
        plot_filename = f"{folder_name}/{month}_bar_plot.png"
        plt.savefig(plot_filename)

        plt.show()
        
     
    
    
    def execute_analysis(self, month):
        correlation_methods = ['pearson', 'kendall', 'spearman', 'mutual']
        correlation_dfs = self.correlations
        features = [5, 10, 15, 20]
        
        for m in month:
            max_corr = np.inf
            max_corr_feature = None
            max_corr_method = None
            max_corr_y_pred = None
            max_corr_name = None
            best_model = None
            
            for f in features:
                for corr_name, correaltion in zip(correlation_methods, correlation_dfs):
                    X_train, X_test, y_train, y_test = self.processing(self.enso_index_file, correaltion, f, m)
                    current_model, _, _ = self.tune_random_forest_regressor(X_train, y_train)
                    correlation_coeff, y_pred = self.model_bag_reg(current_model, X_test, y_test)
                    if correlation_coeff < max_corr:
                        max_corr = correlation_coeff
                        max_corr_feature = f
                        max_corr_method = correaltion
                        max_corr_name = corr_name
                        max_corr_y_pred = y_pred
                        best_model = current_model
            
            print(f"{m}, {max_corr_name} correlation {max_corr_feature} features highest correlation coefficient: {max_corr:.2f}")
            
            self.best_models[f"{m}"] = best_model
            X_train, X_test, y_train, y_test = self.processing(self.enso_index_file, max_corr_method, max_corr_feature, m)
#             best_model, _, _ = self.tune_bagging_regressor(X_train, y_train)
            correlation_coeff, y_pred = self.model_bag_reg(best_model, X_test, y_test)
            self.y_preds[f"{m}"] = y_pred
            self.plotting(y_test, max_corr_y_pred, m, correlation_coeff, max_corr_feature, max_corr_name)
            self.plotting_y_pred_vs_y_test(y_test, max_corr_y_pred, m)
        return self.best_models,self.scalers,self.y_preds,self.y_tests
        
        
        

        




# In[19]:


# nc_file = "Vwind.nc"
# enso_index_file = "enso_index.csv"
# analysis = ENSOAnalysis_br(nc_file, enso_index_file)
# analysis.execute_analysis(['Avg', 'march', 'april', 'may']) 

