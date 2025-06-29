for name in inconsistence_analysis['names']:
    drug_ids_found = dataset[dataset["DRUG_NAME_x"].str.upper() == name.upper()]["DRUG_ID"].unique()
    companies_ids_found = dataset[dataset["DRUG_NAME_x"].str.upper() == name.upper()]["COMPANY_ID"].unique()
    print(f"-- For the name: {name:<12} ---> we detect this ids: {drug_ids_found} and this drug has as its supplier: {companies_ids_found} ")
print(f"Into the company column there are {dataset['COMPANY_ID'].isna().sum()} null values")