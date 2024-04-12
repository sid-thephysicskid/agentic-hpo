import pandas as pd

def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

def data_report(df, num_feats, bin_feats, nom_feats):
   
    # Last column is the label
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # General dataset info
    num_instances = len(df)
    num_features = features.shape[1]

    # Label class analysis
    class_counts = target.value_counts()
    class_distribution = class_counts/num_instances
    if any(class_distribution<0.3) or any(class_distribution>0.7):
        class_imbalance = True
    else:
        class_imbalance = False
   
    # Create a text report
    report = f"""Data Characteristics Report:

- General information:
  - Number of Instances: {num_instances}
  - Number of Features: {num_features}

- Class distribution analysis:
  - Class Distribution: {class_distribution.to_string()}
  {'Warning: Class imbalance detected.' if class_imbalance else ''}

- Feature analysis:
  - Feature names: {features.columns.to_list()}
  - Number of numerical features: {len(num_feats)}
  - Number of binary features: {len(bin_feats)}
  - Binary feature names: {bin_feats}
  - Number of nominal features: {len(nom_feats)}
  - Nominal feature names: {nom_feats}
"""
   
    return report