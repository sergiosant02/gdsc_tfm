def represent_scatter_matrix(ax, y_test_df, predition_values, target_variable, color_scatter="black"):
    if type(color_scatter) is str:
        ax.scatter(np.array(y_test_df[target_variable]), predition_values, s=2, alpha=0.5, color=color_scatter)
    else:
        ax.scatter(np.array(y_test_df[target_variable]), predition_values, s=2, alpha=0.5, c=color_scatter, cmap='tab10')
        legend_labels = np.unique(color_scatter)
        cmap = plt.get_cmap('tab10')
        legend_labels = np.unique(color_scatter)
        for label in legend_labels:
            color_index = label if label >= 0 else 9  # Handle -1 as "noise" with last color
            color = cmap.colors[color_index % len(cmap.colors)]
            ax.scatter([], [], color=color, label=f"Cluster {label}")
        ax.legend()
    ax.grid(True)

    min_val = min(np.min(y_test_df[target_variable]), np.min(predition_values))
    max_val = max(np.max(y_test_df[target_variable]), np.max(predition_values))
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Pronostics')
    ax.set_xlabel('Real')
    ax.set_title(target_variable)

def show_evaluation(model, X_test, y_test):
    pred_test = model.predict(X_test)
    print(f"Evaluation value with test values: {model.evaluate(X_test, y_test)}")
    print(f"MSE: {mean_squared_error(y_test, pred_test)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_test))}")
    print(f"MAE: {mean_absolute_error(y_test, pred_test)}")
    print(f"R2: {r2_score(y_test, pred_test)}")
    rows = int(np.ceil(len(y_test.columns)/2))
    fig, axes = plt.subplots(rows, 2, figsize=(8 * rows, 8 * rows))
    axes = axes.flatten()
    for i, target in enumerate(y_test.columns):
        represent_scatter_matrix(axes[i], y_test_df=y_test, predition_values=pred_test[:, i], target_variable=target)
    fig.tight_layout()
    plt.show()