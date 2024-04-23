import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, make_scorer
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Hyperparameter Tuning")
    parser.add_argument("file_path", type=str, help="Path to the data file")
    parser.add_argument("target_column", type=str, help="Name of the target column")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of XGBoost trees")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators in XGBoost")
    parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio for XGBoost")

    args = parser.parse_args()

    xgb_cross_validation(
        file_path=args.file_path,
        target_column=args.target_column,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample
    )