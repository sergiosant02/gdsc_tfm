input_shape = (X_train.shape[1],)
inputs = Input(shape=input_shape)

x = layers.Dense(2048, activation="silu")(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Reshape((16, 128))(x)
x = layers.Conv1D(16, kernel_size=(5), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.silu(x)
x = layers.SpatialDropout1D(0.5)(x)
x = layers.AveragePooling1D(pool_size=2)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = keras.activations.silu(x)
x = layers.SpatialDropout1D(0.6)(x)
x_main = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
x_main = keras.activations.silu(x_main)
x_main = layers.SpatialDropout1D(0.5)(x_main)
x_main = layers.Conv1D(16, kernel_size=(3), padding="same", kernel_regularizer=regularizers.l2(1e-4))(x_main)
x_main = keras.activations.silu(x_main)
x_main = layers.SpatialDropout1D(0.6)(x_main)
conc = layers.Multiply()([x, x_main])
x_final = layers.MaxPooling1D(pool_size = 4, strides = 2)(conc)
x_final = layers.Flatten()(x_final)
x_final = layers.Dropout(0.4)(x_final)

out_ln_ic50 = Dense(kbins, activation='softmax', name='LN_IC50')(x_final)
out_auc = Dense(kbins, activation='softmax', name='AUC')(x_final)
out_dr = Dense(kbins, activation='softmax', name='Drug_Response')(x_final)
out_rmse = Dense(kbins, activation='softmax', name='RMSE')(x_final)
out_zscore = Dense(kbins, activation='softmax', name='Z_SCORE')(x_final)

model = Model(inputs=inputs, outputs=[out_ln_ic50, out_auc, out_dr, out_rmse, out_zscore])

model.compile(
    optimizer="adam",
    loss="categorical_focal_crossentropy",
    metrics=["categorical_accuracy","categorical_accuracy","categorical_accuracy","categorical_accuracy","categorical_accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_LN_IC50_loss', mode="min", patience=14,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_LN_IC50_loss', factor=0.2, patience=7, min_lr=0.000001)
model_checkpoint = ModelCheckpoint("model_conv_softmax_2.keras", monitor="val_LN_IC50_loss", mode='min')

history = model.fit(
    X_train,
    y_train_dict_onehot,
    validation_data=(X_val, y_val_dict_onehot),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, PlotLossesKerasTF(), reduce_lr, model_checkpoint]
)