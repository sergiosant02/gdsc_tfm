def generate_model(n_columns_x, n_columns_y):
    model = models.Sequential()
    model.add(layers.Input(shape=(n_columns_x,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_columns_y))
    return model