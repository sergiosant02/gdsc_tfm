imputer = KNNImputer(n_neighbors=8) # This number was determined by checking the results obtained from various combinations.
dataset_imputed = imputer.fit_transform(dataset_encoded)