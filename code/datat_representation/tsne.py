if os.path.exists("tsne_3d_result.npy"):
    X_3d_tsne = np.load("tsne_3d_result.npy")
    X_2d_tsne = np.load("tsne_2d_result.npy")
else:
    tsne_3d = TSNE(n_components=3, random_state=10, perplexity=40.0, init='random', learning_rate='auto')
    X_3d_tsne = tsne_3d.fit_transform(X_data)
    np.save("tsne_3d_result.npy", X_3d_tsne)

    tsne_2d = TSNE(n_components=2, random_state=10, perplexity=40.0, init='random', learning_rate='auto')
    X_2d_tsne = tsne_2d.fit_transform(X_data)
    np.save("tsne_2d_result.npy", X_2d_tsne)
    
fig = plt.figure(figsize=(12,7))
ax_1 = fig.add_subplot(121)
ax_1.set_title("2D representation (t-TSNE)")
sc = ax_1.scatter(X_2d_tsne[:, 0], X_2d_tsne[:, 1], c=y_data["LN_IC50"], cmap='viridis', alpha=0.8, s=s)

ax_2 = fig.add_subplot(122, projection="3d")
ax_2.set_title("3D representation (t-TSNE)")
sc = ax_2.scatter(X_3d_tsne[:, 0], X_3d_tsne[:, 1], X_3d_tsne[:, 2], c=y_data["LN_IC50"], cmap='viridis', alpha=0.8, s=s)