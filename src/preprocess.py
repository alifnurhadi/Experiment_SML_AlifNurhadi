import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler


def preprocess_data(input_df, target_col: str, extra_drop_cols: str = ""):
    """
    # Processing data for mlflow

    #### input_df : dataframe to be proceed.
    #### target_col : column name that are being proceed as target.
    #### extra_drop_cols : column name that are being dropped.

    """
    input_df = input_df.drop_duplicates()
    input_df = input_df.dropna()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Identify categorical and numerical columns
    y = input_df[[target_col]]
    X = input_df.drop(
        [extra_drop_cols, target_col] if extra_drop_cols != "" else [target_col],
        axis=1,
    )

    # Train-Test Split (80% Train, 20% Test, Stratified)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    categ_cols = x_train.select_dtypes(include=["object"]).columns
    num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns

    if len(num_cols) > 0:
        input_df[num_cols] = num_imputer.fit_transform(input_df[num_cols])
    if len(categ_cols) > 0:
        input_df[categ_cols] = cat_imputer.fit_transform(input_df[categ_cols])

    bin_cols = ["Weekly_GenAI_Hours", "Traditional_Study_Hours"]
    num_cols = [
        c
        for c in x_train.select_dtypes(include=["int64", "float64"]).columns
        if c not in bin_cols
    ]
    hierarchical_cols = [
        "Year_of_Study",
        "Prompt_Engineering_Skill",
        "Institutional_Policy",
    ]

    ohe_cols = []
    le_cols = []

    for col in categ_cols:
        unique_count = x_train[col].nunique()
        if col not in hierarchical_cols and unique_count <= 5:
            ohe_cols.append(col)
        else:
            le_cols.append(col)

        x_train_encoded = x_train.copy()
        x_test_encoded = x_test.copy()

        # binning data
        for col in bin_cols:
            train_binned, bins = pd.qcut(
                x_train[col], q=5, retbins=True, duplicates="drop"
            )

            bins[0] = -np.inf
            bins[-1] = np.inf

            test_binned = pd.cut(x_test[col], bins=bins, include_lowest=True)

            le_bin = LabelEncoder()
            x_train_encoded[col] = le_bin.fit_transform(train_binned.astype(str))
            x_test_encoded[col] = le_bin.transform(test_binned.astype(str))

        label_encoders = {}
        for col in le_cols:
            le = LabelEncoder()
            x_train_encoded[col] = le.fit_transform(x_train_encoded[col])
            x_test_encoded[col] = le.transform(x_test_encoded[col])
            label_encoders[col] = le

        # One-Hot Encoding
        if ohe_cols:
            x_train_ohe_part = pd.get_dummies(
                x_train_encoded[ohe_cols], columns=ohe_cols, drop_first=True
            )
            x_test_ohe_part = pd.get_dummies(
                x_test_encoded[ohe_cols], columns=ohe_cols, drop_first=True
            )

            x_test_ohe_part = x_test_ohe_part.reindex(
                columns=x_train_ohe_part.columns, fill_value=0
            )

            x_train_encoded = pd.concat(
                [x_train_encoded.drop(ohe_cols, axis=1), x_train_ohe_part], axis=1
            )
            x_test_encoded = pd.concat(
                [x_test_encoded.drop(ohe_cols, axis=1), x_test_ohe_part], axis=1
            )

        std_scaler = StandardScaler()
        rob_scaler = RobustScaler()

        skewed_cols = []
        normal_cols = []

        # scalling based result of check skewness on train set
        for col in num_cols:
            skew_val = x_train[col].skew()
            if abs(skew_val) > 1.0:
                skewed_cols.append(col)
                x_train_encoded[col] = rob_scaler.fit_transform(x_train_encoded[[col]])
                x_test_encoded[col] = rob_scaler.transform(x_test_encoded[[col]])
            else:
                normal_cols.append(col)
                x_train_encoded[col] = std_scaler.fit_transform(x_train_encoded[[col]])
                x_test_encoded[col] = std_scaler.transform(x_test_encoded[[col]])

    return x_train_encoded, x_test_encoded, y_train_encoded, y_test_encoded
