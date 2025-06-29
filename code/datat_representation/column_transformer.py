yes_no_map = {"N": -1, np.nan: 0, "Y": 1}
adhenrence_map = {"Adherent": 2, "Semi-Adherent": 1, "Suspension": 0}
screen_medium = {'R': 0, 'D/F12': 1}

class MapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping):
        self.column = column
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.column] = X_[self.column].map(self.mapping).astype(np.int64)
        return X_
    
column_pipeline = ColumnTransformer(
    transformers=[ 
        ('map_methylation', MapTransformer(column="Methylation", mapping=yes_no_map), ["Methylation"]),
        ('map_response', MapTransformer(column='Drug Response', mapping=yes_no_map), ['Drug Response']),
        ('map_cna', MapTransformer(column="Copy Number Alterations (CNA)", mapping=yes_no_map), ['Copy Number Alterations (CNA)']),
        ('map_expression', MapTransformer(column="Gene Expression", mapping=yes_no_map), ['Gene Expression']),
        ("map_adherence", MapTransformer(column="Growth Properties", mapping=adhenrence_map), ["Growth Properties"]),
        ("map_screen_medium", MapTransformer(column="Screen Medium", mapping=screen_medium), ["Screen Medium"])
    ],
    remainder="passthrough"
)

transformed_cols = ["Methylation", "Drug Response", "Copy Number Alterations (CNA)", "Gene Expression", "Growth Properties", "Screen Medium"]
remainder_cols = [col for col in dataset.columns if col not in transformed_cols]
column_names = transformed_cols + remainder_cols

dataset_copy = dataset.copy()
dataset_data_preprocessed = column_pipeline.fit_transform(dataset)
dataset_preprocessed = pd.DataFrame(dataset_data_preprocessed, columns=column_names)

for col in dataset_preprocessed.columns:
    if col in transformed_cols:
        dataset_preprocessed[col] = dataset_preprocessed[col].astype(int)
    else:
        dataset_preprocessed[col] = dataset[col].copy()