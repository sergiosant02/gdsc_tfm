columns_to_impute = ["TCGA_DESC", "PUTATIVE_TARGET", "Cancer Type\n(matching TCGA label)", "Microsatellite \ninstability Status (MSI)", "TARGET"]
encoder_nan = OrdinalEncoder()
encoder_all = OrdinalEncoder()
dataset_encoded[columns_to_impute] = encoder_nan.fit_transform(dataset[columns_to_impute])
columns_not_encode = [i for i in categorical_variables if i not in columns_to_impute]
dataset_encoded[columns_not_encode] = encoder_all.fit_transform(dataset[columns_not_encode].astype(str))