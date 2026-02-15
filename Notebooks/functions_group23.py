########## Imports ############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rapidfuzz import process, fuzz
from tqdm import tqdm
from IPython.display import display # Necessário para a função dict_to_results_df funcionar num script
from tqdm_joblib import tqdm_joblib

########## Functions ##########



def strange_values_to_nan(f, column, limit):
    """
    Replaces values in a column that are less than or equal to a limit with NaN.

    Arguments:
        f (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to check.
        limit (float): The threshold value. Values <= this limit will be replaced with NaN.

    Output:
        None: The function modifies the DataFrame 'f' in-place.
    """
    # Create a boolean mask where True indicates "strange" values (<= limit)
    strange_values = f[column] < limit
    
    # Replace the identified values with NaN
    f.loc[strange_values, column] = np.nan
    
    # Print the number of found strange values
    print(f"Found {strange_values.sum()} strange values in {column}.")




def pre_processing_metric(df):
    """
    Performs initial preprocessing of metric columns: index setting, type fixing, 
    and cleaning strange values.

    Arguments:
        df (pd.DataFrame): The DataFrame to process.

    Output:
        None: Modifies the DataFrame in-place.
    """
    # Setting carID as index
    df.set_index("carID", inplace=True)

    # Fixing the types: convert to int if float/valid, then cast to nullable Int64
    df["year"] = [int(i) if isinstance(i, float) and not pd.isna(i) else i for i in df["year"]]
    df["year"] = df["year"].astype("Int64")
    
    df["previousOwners"] = [int(i) if isinstance(i, float) and not pd.isna(i) else i for i in df["previousOwners"]]
    df["previousOwners"] = df["previousOwners"].astype("Int64")

    # Converting strange values to nan using the helper function
    strange_values_to_nan(df, "mileage", 0)
    strange_values_to_nan(df, "tax", 0)
    strange_values_to_nan(df, "mpg", 10)
    strange_values_to_nan(df, "engineSize", 0.1)
    strange_values_to_nan(df, "previousOwners", 0)
    strange_year = df["year"] > 2020
    df.loc[strange_year, "year"] = np.nan

    # Deleting column hasDamage because it has the same value in all rows
    del df["hasDamage"]
    
    # Deleting column paintQuality% because the model needs to be capable of evaluating the price 
    # of a car based on the user’s input without needing the car to be taken to a mechanic
    del df["paintQuality%"]




def build_valid_dic(column, valid_list, cuttoff):
    """
    Builds a dictionary mapping valid words to their fuzzy matches found in a column.

    Arguments:
        column (pd.Series): The data column to search in.
        valid_list (list): List of valid strings to match against.
        cuttoff (int): Minimum score for a match to be considered valid.

    Output:
        dict: A dictionary where keys are valid words and values are lists of found variations.
    """
    # First, clean the column and the valid list (lowercase, stripped)
    unique_column_values = [str(v).strip().lower() for v in column.dropna().unique()]
    
    # Second, create the dict with the valid values initialized with empty lists
    result = {valid_word: [] for valid_word in valid_list}

    for valid in valid_list:
        valid_lower = valid.lower()
        # Use rapidfuzz to find all strings in the column that are similar to the valid word
        fuzzy_matched_values = process.extract(
            valid_lower,           # the check value
            unique_column_values,  # the list that we are going to use to check
            scorer=fuzz.token_sort_ratio,  # Use a scorer that handles out-of-order words
            score_cutoff=cuttoff   # Minimum similarity score
        )
        for match_value, score, _ in fuzzy_matched_values:
            # Add the matched value to our dictionary, but only if it's not an exact match
            if match_value != valid_lower:
                result[valid].append(match_value)

    return result




def replace_invalid_values(column, valid_dic):
    """
    Replaces invalid values in a column based on a mapping dictionary.

    Arguments:
        column (pd.Series): The column to correct.
        valid_dic (dict): Dictionary mapping valid terms to their invalid variations.

    Output:
        list: A list of corrected values.
    """
    corrected = []
    for val in column:
        if pd.isna(val):
            corrected.append(val)
            continue
            
        valeu_cleaned = str(val).strip().lower()
        found = False

        # Check for an exact match (case-insensitive)
        for key in valid_dic:
            if valeu_cleaned == key.lower():
                corrected.append(key)
                found = True
                break
                
        # Check for a fuzzy match in the variations list
        if not found:
            for valid_key, variations in valid_dic.items():
                # Check if the cleaned value is one of the known variations
                if valeu_cleaned in [v.lower() for v in variations]:
                    corrected.append(valid_key)
                    found = True
                    break
                    
        # If no match found, keep the original value
        if not found:
            corrected.append(val)

    return corrected




def fill_nans_categorical(data, columns):
    """
    Fills NaN values in specified categorical columns with the string 'Unknown'.
    
    Arguments:
        data (pd.DataFrame): The DataFrame.
        columns (list): List of column names to fill.
        
    Output:
        pd.DataFrame: The modified DataFrame.
    """
    # Iterate over the list of categorical columns provided
    for col in columns:
        # Fill missing values (NaN) in the current column with the string "Unknown"
        data[col] = data[col].fillna("Unknown")
    return data




def dic_brand_modles(brands, models, valid_brands):
    """
    Builds a dictionary mapping each valid brand to all unique models associated with it.
    
    Arguments:
        brands (pd.Series): Series containing brand names.
        models (pd.Series): Series containing model names.
        valid_brands (list): List of valid brands to include.
        
    Output:
        dict: Dictionary {brand: [list of models]}.
    """
    # Initialize the result dictionary with valid brands as keys and empty lists as values
    result = {brand: [] for brand in valid_brands}
    
    # Iterate through both brands and models simultaneously
    for brand, model in zip(brands, models):
        # Skip the iteration if either the brand or the model is missing (NaN)
        if pd.isna(brand) or pd.isna(model):
            continue
            
        # Clean the model name: convert to string, remove spaces, and make it lowercase
        model_clean = str(model).strip().lower()
        
        # Check if the cleaned model is not already in the list for that brand
        if model_clean not in result[brand]:
            # Append the new unique model to the brand's list
            result[brand].append(model_clean)

    return result




def pre_processing_non_metric(dic_col_valid, data):
    """
    Iteratively cleans non-metric columns using fuzzy matching.
    
    Arguments:
        dic_col_valid (dict): Dictionary of valid values for each column.
        data (pd.DataFrame): The DataFrame to clean.
        
    Output:
        None: Modifies DataFrame in-place (and prints value counts on last iteration).
    """
    # Loop 3 times for iterative cleaning. 
    # E.g., Pass 1 might turn 'fordd' into 'Ford'.
    # Pass 2 can then use the cleaner list to find more complex variations.
    for i in range(1, 4):
        for key, values in dic_col_valid.items():
            # Find all variations for each valid value
            valid_dic = build_valid_dic(data[key], values, cuttoff=50)
            # Replace those variations in the column
            data[key] = replace_invalid_values(data[key], valid_dic)
            if i == 3:  # On the final pass, print the value counts
                print(data[key].value_counts())





def valid_models_dict(data, brands, min_count=0):
    """
    Creates a dictionary of valid models for each brand, filtering by frequency.
    
    Arguments:
        data (pd.DataFrame): The dataset.
        brands (list): List of brands to process.
        min_count (int): Minimum frequency for a model to be considered valid.
        
    Output:
        dict: Dictionary of valid models per brand.
    """
    models_dict = {}
    for brand in brands:
        brands_frame = data[data["Brand"] == brand]
        models_clean = brands_frame["model"].astype(str).str.strip().str.lower()

        # Get all model names for this brand, sorted by frequency
        models_count = models_clean.value_counts()
        # Filter out very rare models
        models_count = models_count[models_count >= min_count]
        potential_models = models_count.index.tolist()
        valid_models_for_brand = []

        for model in potential_models:
            is_truncated = False
            # This logic checks if a longer, more specific model name already exists.
            # It will be flagged as truncated and skipped.
            for valid_model in valid_models_for_brand:
                if valid_model.startswith(model):
                    is_truncated = True
                    break

            if not is_truncated:
                valid_models_for_brand.append(model)

        models_dict[brand] = valid_models_for_brand

    return models_dict





def fill_unknown_brand(data, valid_models_dict):
    """
    Infers the 'Brand' for rows where Brand is 'Unknown' by checking the 'model' name
    against the master dictionary of valid models per brand.
    
    Arguments:
        data (pd.DataFrame): The DataFrame to process.
        valid_models_dict (dict): Dictionary mapping brands to lists of their models, that was predefined .
        
    Output:
        pd.DataFrame: The DataFrame with imputed brands.
    """
    # Select only rows where Brand is 'Unknown'
    unknown_df = data[data['Brand'] == 'Unknown']
    for idx, row in unknown_df.iterrows():
        model_lower = str(row['model']).lower()
        # Check this model against every brand's list of known models
        for brand, models in valid_models_dict.items():
            if model_lower in [m.lower() for m in models]:
                # If found, update the Brand in the original dataframe
                data.at[idx, 'Brand'] = brand
                break

    return data





def impute_numeric_features(X_features_list, metric_features):
    """
    Imputes missing numeric values using the median strategy.
    
    Arguments:
        X_features_list (list): List of DataFrames [X_train, X_val, ...].
        metric_features (list): List of numeric column names.
        
    Output:
        list: List of DataFrames with imputed values.
    """
    # Initialize the imputer using 'n_neighbors' = 10.
    imputer = KNNImputer(n_neighbors=10)

    # Select the numeric features from the training set (assumed to be at index 0)
    X_train_numeric_to_fit = X_features_list[0][metric_features]

    # Fit the imputer ONLY on the training data to prevent data leakage
    imputer.fit(X_train_numeric_to_fit)

    imputed_dfs_list = []

    # Iterate through all dataframes (train, val, test) to transform them
    for df in X_features_list:
        # Create a copy to avoid modifying the original list in place
        df_copy = df.copy()

        # Transform the data using the fitted imputer (replace NaNs with median)
        imputed_data_np = imputer.transform(df_copy[metric_features])

        # Assign the imputed values back to the specific columns
        df_copy[metric_features] = imputed_data_np

        imputed_dfs_list.append(df_copy)
    return imputed_dfs_list





def scale_numeric_features(X_features, metric_features):
    """
    Scales numeric features using RobustScaler.
    
    Arguments:
        X_features (list): List of DataFrames [X_train, X_val, ...].
        metric_features (list): List of numeric column names.
        
    Output:
        list: List of scaled DataFrames.
    """
    # Initialize the RobustScaler (good for data with outliers)
    scaler = RobustScaler()

    # Create a copy of the training set (index 0) to fit the scaler
    X_train = X_features[0].copy()
    
    # Fit the scaler using only the training data
    scaler.fit(X_train[metric_features])

    scaled_features_list = []

    # Transform all datasets (train, val, test) using the parameters learned from train
    for i, X, in enumerate(X_features):
        X_scaled = X.copy()
        # Apply the transformation
        X_scaled[metric_features] = scaler.transform(X[metric_features])
        scaled_features_list.append(X_scaled)

    return scaled_features_list

#-----------
# feature selection functions


def _get_lasso(X_num, y):
    """Calculates Lasso regression coefficients to identify important numeric features.

    Arguments:
        X_num (pd.DataFrame): DataFrame containing only numeric features.
        y (pd.Series): The target variable.

    Output:
        pd.Series: Absolute values of the coefficients for each feature.
    """

    lasso = LassoCV(cv=5, random_state=42).fit(X_num, y)
    return pd.Series(np.abs(lasso.coef_), index=X_num.columns) # returns absolute value since magnitude matters more than the signal

def _get_rfe(X_num, y):
    """Calculates RFE ranking for numeric features.

    Arguments:
        X_num (pd.DataFrame): DataFrame containing only numeric features.
        y (pd.Series): The target variable.

    Output:
        pd.Series: The ranking of features (1 = Most Important, higher numbers = Less Important).
    """

    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=1) # Força a rankear tudo
    rfe.fit(X_num, y)

    # in this output, lower is better
    # we will invert this later in the main function to match other metrics (where higher is better).
    return pd.Series(rfe.ranking_, index=X_num.columns)

# decision trees

def _get_model_importance(model_class, X_encoded, y):
    """
    Generic helper function to train a tree-based model and extract feature importance.

    Arguments:
        model_class (class): The Scikit-Learn model class (e.g., RandomForestRegressor).
        X_encoded (pd.DataFrame): The input features (must be fully numeric/encoded).
        y (pd.Series): The target variable.

    Output:
        pd.Series: The feature importance scores extracted from the trained model.
    """
   
    model = model_class(random_state=42)
    
    # checks if the model supports parallel processing (GradientBoostingRegressor doesnt)
    if 'n_jobs' in model.get_params():
        model.set_params(n_jobs=-1) # n_jobs=-1 uses all available CPU cores for faster training
        
    model.fit(X_encoded, y)
    
    return pd.Series(model.feature_importances_, index=X_encoded.columns)

# calculating importance for linear models and decision trees

def calculate_importances(X_num, X_cat, y):
    """
    Runs multiple models (Linear and Tree-based), aggregates the importance scores, handles specific preprocessing for trees,
    and returns a normalized DataFrame for easy comparison.

    Arguments:
        X_num (pd.DataFrame): Numeric features (already scaled).
        X_cat (pd.DataFrame): Categorical features.
        y (pd.Series): Target variable.

    Output:
        pd.DataFrame: A normalized (0-1) table of feature importances across all models.
    """
    results = {}

    # linear methods (for numeric features)
    # we exclude categorical features from linear method because:
    # - if we apply ordinal encoding to a nominal category, the model incorrectly assumes a mathematical relationship
    # - if we apply one hot encoding it creates lots of one-hot columns 
    # so we use tree models to evaluate categories
    if not X_num.empty:
        results['Lasso'] = _get_lasso(X_num, y)
        results['RFE'] = _get_rfe(X_num, y)
    
    # tree models (for numeric and categorical features)
    # tree-based models dont perform mathematical operations on features
    # they make logical splits, so we can use ordinal encoding
    X_all = pd.concat([X_num, X_cat], axis=1)
    
    text_cols = X_all.select_dtypes(include=['object', 'category']).columns
    if len(text_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_all[text_cols] = encoder.fit_transform(X_all[text_cols].astype(str))

    # calls the helper function we defined earlier
    results['RandomForest'] = _get_model_importance(RandomForestRegressor, X_all, y)
    results['ExtraTrees'] = _get_model_importance(ExtraTreesRegressor, X_all, y)
    results['GradBoosting'] = _get_model_importance(GradientBoostingRegressor, X_all, y)

    df = pd.DataFrame(results).fillna(0)
    
    # inverts RFE Ranking so that RFE gives 1 for the best feature
    if 'RFE' in df.columns:
        df['RFE'] = 1 / (df['RFE'].replace(0, 1))

    return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)

# plots the linear analysis and non linear

def plot_dashboard_strategy(df_norm, corr_matrix, numeric_features_list):
    """
    Generates a dashboard comprising two main figures.

    Figure 1 (Linear Analysis):
        - Left panel: Spearman Correlation Matrix to inspect multicollinearity between all numeric features.
        - Right panel: Feature Importance from Linear Models (Lasso & RFE), applied only to numeric features.
    
    Figure 2 (Non-Linear):
        - Bottom panel: A consolidated view of feature importance derived from Tree-based models 
          (Random Forest, Extra Trees, Gradient Boosting). 

    Arguments:
        df_norm (pd.DataFrame): DataFrame containing normalized importance scores (0-1) for all models.
        corr_matrix (pd.DataFrame): Pre-calculated Spearman correlation matrix (All vs All).
        numeric_features_list (list): List of names of numeric features (used to filter the linear plot).
    """

    # figure 1
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1.2, 1]}) #gives more horizontal space to spearman
    
    # left plot
    # correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.2f', ax=axes[0], cbar_kws={'label': 'Spearman Correlation'})
    
    axes[0].set_title("1. Correlation matrix (Spearman)", fontsize=14, fontweight='bold')
    
    # right plot
    # rfe and lasso plots
    linear_methods = ['Lasso', 'RFE']
    valid_num = [f for f in numeric_features_list if f in df_norm.index]
    
    if valid_num:
        # selects rows corresponding to numeric features and columns
        df_lin = df_norm.loc[valid_num, linear_methods]
        
        if not df_lin.empty:
            # calculates average importance for sorting 
            df_lin['Avg'] = df_lin.mean(axis=1)
            df_lin_sorted = df_lin.sort_values(by='Avg', ascending=True).drop(columns=['Avg'])
            
            palette_lin = ['#1f77b4', '#d62728'] # blue, red
            
            # plot on the right axis 
            df_lin_sorted.plot(kind='barh', width=0.8, color=palette_lin, ax=axes[1])
            
            axes[1].set_title("2. Linear feature importance (Lasso and RFE)", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Normalized importance (0 to 1)")
            axes[1].grid(axis='x', linestyle='--', alpha=0.5)
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    plt.show()

    # figure 2
    # tree models
    # from the 5 models that we used later on stacking:
    # - we selected RandomForest, ExtraTrees, and GradientBoosting because they all share 
    # the .feature_importances_ metric (this garantees that the resulting scores in the bar charts are comparable)
    # - we didnt select HistGradientBoosting because it doesent have the .feature_importances_ attribute and KNN because
    # it uses geometric distance, not tree splits, that is not directly comparable here
    tree_methods = ['RandomForest', 'ExtraTrees', 'GradBoosting']
    tree_cols = [c for c in tree_methods if c in df_norm.columns]
    
    if tree_cols:
        df_trees = df_norm[tree_cols].copy()
        # calculates the average importance across the three tree models to sort
        df_trees['Avg'] = df_trees.mean(axis=1)
        df_trees_sorted = df_trees.sort_values(by='Avg', ascending=True).drop(columns=['Avg']).tail(15)
        
        my_palette = ['#1f77b4', '#ff7f0e', '#2ca02c'] # blue, orange, green
        
        plt.figure(figsize=(14, 10))
        df_trees_sorted.plot(kind='barh', width=0.92, color=my_palette, figsize=(14, 10))
        
        plt.title("3. Non-linear feature importance (tree-based)", fontsize=16, fontweight='bold')
        plt.xlabel("Normalized importance (0 to 1)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
#----


def running_models(models, X_train, y_train, X_val, y_val):
    """
    Trains and evaluates models defined in the global 'models' dictionary.
    
    Arguments:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        
    Output:
        None: Updates the global 'results' dictionary and prints metrics.
    """
    # Iterate over the global dictionary of models

    results = {}
    for name, model in models.items():
        # Train the model using the training data
        model.fit(X_train, y_train)

        # Generate predictions using the validation set
        y_pred_val = model.predict(X_val)
        
        # Calculate performance metrics
        r2 = r2_score(y_val, y_pred_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        
        # Store the results in the global 'results' dictionary
        results[name] = {"R²": r2, "MAE": mae, "RMSE": rmse,
                         ##"pred": y_pred_val
                         }
        
        # Print the metrics for the current model
        print(f"\nResults for: {name}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")

    return results





def dict_to_results_df(results_dict, sort_by="R²", ascending=False, display_df=False):
    """
    Converts the results dictionary to a DataFrame for visualization.
    
    Arguments:
        results_dict (dict): Dictionary of results.
        sort_by (str): Column to sort by.
        ascending (bool): Sort order.
        display_df (bool): Whether to display the DataFrame.
        
    Output:
        pd.DataFrame: The results table.
    """
    # Convert dictionary to DataFrame, transpose it, and reset index to make model names a column
    df = pd.DataFrame(results_dict).T.reset_index().rename(columns={"index": "Model"})

    # Define rounding precision for each metric
    rounding = {"R²": 5, "MAE": 2, "RMSE": 2}

    # Apply rounding to the columns if they exist in the DataFrame
    for col, digits in rounding.items():
        if col in df.columns:
            df[col] = df[col].round(digits)

    # Sort the DataFrame based on the specified column
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # Optionally display the DataFrame (useful in Jupyter Notebooks)
    if display_df:
        display(df)

    return df





def tune_models(models_params, X_train, y_train, cv, scoring='neg_mean_absolute_error', n_iter=50):
    """
    Receives a dictionary with models and their parameters, performs RandomizedSearchCV, 
    and returns a dictionary with the best trained models.
    Shows progress bar with tqdm.

    Arguments:
        models_params (dict): Dict of models and params.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv (int): Cross-validation splits.
        scoring (str): Scoring metric.
        n_iter (int): Number of iterations.

    Output:
        dict: Best models found.
    """
    best_models = {}
    n_splits = cv.get_n_splits()
    
    # Iterate through the models and their parameter grids
    for name, (model, param_grid) in tqdm(models_params.items(), desc="Training models", unit="model"):
        # Configure the Randomized Search Cross-Validation
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        total_fits = n_iter * n_splits  # Calculate total fits for progress tracking

        # Fit the search algorithm to the training data
        with tqdm_joblib(tqdm(total=total_fits, desc=f"Random Search {name}", leave=False)):
            search.fit(X_train, y_train)

        # Print the best score (negated because scoring is negative MAE) and best parameters
        print(f"Melhor score {name}: {-search.best_score_:.4f}")
        print(f"Melhor parâmetros: {search.best_params_}")

        # Store the fitted search object in the dictionary
        best_models[name] = search

    return best_models



def create_new_features(df):

    current_year = 2020
    df['log_mileage'] = np.log1p(df['mileage'])

    df['car_age'] = current_year - df['year']
    df['age_squared'] = df['car_age'] ** 2
    df['miles_per_year'] = df['log_mileage'] / (df['car_age'] + 0.1)

    df['power_efficiency'] = df['engineSize'] / (df['mpg'] + 1)

    return df