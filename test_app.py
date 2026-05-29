import sys
sys.path.insert(0,'.')

print("Testing data loader...")
from src.data_loader import load_raw_data, preprocess_data, get_eda_stats
df = load_raw_data()
print(f"  Dataset: {df.shape}")
result = preprocess_data(df)
X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, feat_names, scaler, df_enc = result
print(f"  Train/Test: {X_tr.shape} / {X_te.shape}")
print(f"  Features: {len(feat_names)}")

print("\nTraining all models...")
from src.models import train_all_models
results = train_all_models(X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, feat_names)
for name, r in results.items():
    print(f"  {name}: Acc={r['accuracy']}% F1={r['f1']}% AUC={r['auc']}%")

print("\nTesting clustering...")
import numpy as np
from src.clustering import run_pca_kmeans, get_cluster_summary
X_all = np.vstack([X_tr_sc, X_te_sc])
X_pca, labels, pca_m, km, ev = run_pca_kmeans(X_all)
print(f"  PCA: {X_pca.shape}, Clusters: {set(labels)}, Variance: {ev[0]*100:.1f}%+{ev[1]*100:.1f}%")

print("\nTesting prediction...")
from src.models import predict_employee
emp = X_tr[0]
pred = predict_employee(results, scaler, feat_names, emp)
print(f"  Ensemble prob: {pred['ensemble_prob']}%, Risk: {pred['risk_level']}")

print("\nALL TESTS PASSED!")
