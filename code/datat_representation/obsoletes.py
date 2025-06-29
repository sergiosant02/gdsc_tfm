obsolete = []
for col in dataset.columns:
    if col.upper() != ("cosmic_id".upper()):
        content = dataset[col].unique()
        if len(content) < 2:
            obsolete.append(col)
        suffix = "..." if len(content) > 10 else ""
        print(f"\033[1m* {col}\033[0m -> {content[:10]} {suffix}")
print(f"The obsolete columns are: {obsolete}")

dataset = dataset.drop(columns=obsolete)