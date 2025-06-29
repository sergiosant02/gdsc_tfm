def train_sequential_model(X_train, y_train, X_val, y_val, X_test, y_test, model: models.Sequential, optimizer="adam", error="mse", metric=keras.metrics.RootMeanSquaredError()):
    model.compile(optimizer=optimizer, loss=error, metrics=[metric])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("best_model_v1.keras", monitor="val_loss", mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.000001)
    history=model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=32, 
                  validation_data=(X_val, y_val), 
                  callbacks=[early_stopping, model_checkpoint, PlotLossesKerasTF(), reduce_lr],
                  verbose=2)
    show_evaluation(model, X_test, y_test)
    return model