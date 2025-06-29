def check_id_name_consistency(df, id_col="DRUG_ID", name_col="DRUG_NAME_x"):
    names = df[name_col].unique()
    ids_values = df[id_col].unique()
    names_inconsistences = []
    ids_inconsistences = []
    for i in names:
        ids = df[df[name_col] == i][id_col].unique()
        if len(ids) > 1:
            print(f"names' inconsistencies: {i}")
            names_inconsistences.append(i)
    for i in ids_values:
        names = df[df[id_col] == i][name_col].unique()
        if len(names) > 1:
            print(f"ids' inconsistencies: {i}")
            ids_inconsistences.append(i)
    if len(names_inconsistences) > 0:
        print(f"There are inconsistencies in names: {names_inconsistences}")
    if len(ids_inconsistences) > 0:
        print(f"There are inconsistencies in ids: {ids_inconsistences}")
    return {"names": names_inconsistences, "ids": ids_inconsistences}
        
inconsistence_analysis = check_id_name_consistency(dataset)