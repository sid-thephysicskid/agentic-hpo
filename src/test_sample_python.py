import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, make_scorer

def xgb_cross_validation(file_path, target_column, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1):
    # Load data from CSV
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Encode categorical variables if any
    X_encoded = pd.get_dummies(X)
    
    # Encode labels numerically
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Define the model with dynamic hyperparameters
    model = xgb.XGBClassifier(learning_rate=learning_rate, 
                              max_depth=max_depth, 
                              n_estimators=n_estimators, 
                              subsample=subsample,
                              use_label_encoder=False,
                              eval_metric='mlogloss')
    
    # Define F1 score for multi-class classification (average might be 'micro', 'macro', or 'weighted')
    f1 = make_scorer(f1_score, average='macro')
    
    # Perform cross-validation using F1 as the scoring metric
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring=f1)  # 5-fold cross-validation

    # Print the F1 scores
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Example usage:
xgb_cross_validation('path_to_your_data.csv', 'target_column_name', learning_rate=0.2, max_depth=4, n_estimators=150, subsample=0.8)
