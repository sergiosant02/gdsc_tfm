inputs = keras.Input((X_train.shape[1],))

x = layers.Dense(2048, activation="silu")(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Reshape((16, 128))(x)
x = layers.Conv1D(16, kernel_size=(5), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.silu(x)
x = layers.SpatialDropout1D(0.3)(x)
x = layers.AveragePooling1D(pool_size=2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.silu(x)
x = layers.SpatialDropout1D(0.3)(x)
x_main = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x_main = keras.activations.silu(x_main)
x_main = layers.SpatialDropout1D(0.3)(x_main)
x_main = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x_main)
x_main = keras.activations.silu(x_main)
x_main = layers.SpatialDropout1D(0.3)(x_main)
conc = layers.Multiply()([x, x_main])
x_final = layers.MaxPooling1D(pool_size = 4, strides = 2)(conc)
x_final = layers.Flatten()(x_final)
x_final = layers.Dropout(0.3)(x_final)
outputs = layers.Dense(y_test.shape[1])(x_final)

model_conv1d_v25 = keras.Model(inputs=inputs, outputs=outputs, name="convmodelv25")

model_conv1d_v25.compile(optimizer="adam", loss=keras.losses.LogCosh(), metrics=[keras.metrics.RootMeanSquaredError()])
model_conv1d_v25.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',restore_best_weights=True)
model_checkpoint = ModelCheckpoint("model_conv_25.keras", monitor="val_loss", mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.000001)

model_conv1d_v25.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=32, 
                  validation_data=(X_val, y_val), 
                  callbacks=[early_stopping, model_checkpoint, PlotLossesKerasTF(), reduce_lr],
                  verbose=2)
show_evaluation(model_conv1d_v25, X_test, y_test)