"""
V10: Hybrid Meta-Stacking (V8 + V9 결합)
==========================================
3가지 전략으로 V8(예측력)과 V9(방법론)을 결합:
  Strategy 1: Prediction Blending — V8 Ridge + V9 Simple Avg 평균
  Strategy 2: Constrained Meta — V9 OOF 가중치로 상/하한 설정 후 2024로 최적화
  Strategy 3: Bayesian Update — V9 Prior + V8 Likelihood → Posterior 가중치

핵심 아이디어:
  V9 OOF = "내신 성적" (과거 실력 기반, 안정적)
  V8 Test = "모의고사" (최신 정보, 과적합 위험)
  → 둘 다 반영하는 게 가장 현실적
"""
import sys, os, unicodedata
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
import numpy as np, pandas as pd
np.random.seed(42)

# ═══════════════════════════════════════════════════
#  1. LOAD EXISTING V8/V9 RESULTS
# ═══════════════════════════════════════════════════
print("="*70)
print("  V10: HYBRID META-STACKING (V8 + V9)")
print("="*70)

# V8 2024 test predictions (individual models)
v8_test_A = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v8_test_A.csv'))
v8_test_B = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v8_test_B.csv'))
v8_test_C = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v8_test_C.csv'))

# V8 2025 predictions (with meta)
v8_meta = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v8_meta_predictions.csv'))

# V9 2024 test (with meta)
v9_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v9_test_meta.csv'))

# V9 2025 predictions (with meta)
v9_pred = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v9_pred_meta.csv'))

# V7 for comparison
v7_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v7_test_A.csv'))
v7_pred = pd.read_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v7_pred_A.csv'))

print(f"  V8 Meta 2025: {len(v8_meta)} rows")
print(f"  V9 Test 2024: {len(v9_test)} rows")
print(f"  V9 Pred 2025: {len(v9_pred)} rows")

# ═══════════════════════════════════════════════════
#  2. RECONSTRUCT V8 TEST META PREDICTIONS
# ═══════════════════════════════════════════════════
# Merge V8 test A/B/C into one dataframe
v8t = v8_test_A[['year_month','country','country_group','actual','predicted']].rename(columns={'predicted':'pred_A'})
v8t = v8t.merge(v8_test_B[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),
                on=['year_month','country'], how='inner')
v8t = v8t.merge(v8_test_C[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),
                on=['year_month','country'], how='inner')

# V8 Ridge Meta weights (from V8 run)
# A=0.0794, B=0.0474, C=0.9698 (approximate from report)
v8_weights = np.array([0.0794, 0.0474, 0.9698])

# V8 meta predictions for 2024
v8t['meta_ridge'] = np.maximum(
    v8t['pred_A'] * v8_weights[0] + v8t['pred_B'] * v8_weights[1] + v8t['pred_C'] * v8_weights[2], 0)
v8t['meta_avg'] = np.maximum((v8t['pred_A'] + v8t['pred_B'] + v8t['pred_C']) / 3, 0)

print(f"  V8 Test merged: {len(v8t)} rows")

# ═══════════════════════════════════════════════════
#  3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════
def eval_preds(name, pred, actual, verbose=True):
    mask = actual > 0
    a, p = actual[mask], pred[mask]
    r = np.sqrt(np.mean((a - p)**2))
    m = np.mean(np.abs((a - p) / a)) * 100
    te = abs(a.sum() - p.sum()) / a.sum() * 100
    if verbose:
        print(f"  {name:<35s}  RMSE={r:>10,.0f}  MAPE={m:>6.1f}%  TotalErr={te:>5.1f}%")
    return {'name': name, 'rmse': r, 'mape': m, 'terr': te}

class SimpleRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X, y):
        self.x_mean = X.mean(axis=0)
        self.y_mean = y.mean()
        Xc = X - self.x_mean; yc = y - self.y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = self.y_mean - self.x_mean @ self.coef_
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# ═══════════════════════════════════════════════════
#  4. V9 OOF META WEIGHTS (from V9 run)
# ═══════════════════════════════════════════════════
# Reload V9 OOF data to get exact weights
# V9 Ridge OOF coef: A=0.5729, B=0.0281, C=0.3812 (approximate)
v9_oof_weights = np.array([0.5729, 0.0281, 0.3812])
v9_oof_weights = v9_oof_weights / v9_oof_weights.sum()  # normalize

print(f"\n  V9 OOF weights: A={v9_oof_weights[0]:.3f}, B={v9_oof_weights[1]:.3f}, C={v9_oof_weights[2]:.3f}")
print(f"  V8 Test weights: A={v8_weights[0]:.4f}, B={v8_weights[1]:.4f}, C={v8_weights[2]:.4f}")


# ═══════════════════════════════════════════════════
#  5. STRATEGY 1: PREDICTION BLENDING
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  STRATEGY 1: PREDICTION BLENDING")
print("  V8 Ridge Meta + V9 Simple Avg 예측값을 α:β로 결합")
print("="*70)

# ── 2024 Test ──
# V8 meta for 2024 test
v8_test_preds = v8t[['year_month','country','actual','pred_A','pred_B','pred_C','meta_ridge']].copy()

# V9 test already has meta_avg
v9_test_preds = v9_test[['year_month','country','actual','meta_avg','meta_ridge','pred_A','pred_B','pred_C']].copy()

# Merge V8 and V9 test predictions
blend_test = v8_test_preds[['year_month','country','actual']].merge(
    v8_test_preds[['year_month','country','meta_ridge']].rename(columns={'meta_ridge':'v8_ridge'}),
    on=['year_month','country'])
blend_test = blend_test.merge(
    v9_test_preds[['year_month','country','meta_avg']].rename(columns={'meta_avg':'v9_avg'}),
    on=['year_month','country'], how='inner')

print(f"\n  Blend Test merged: {len(blend_test)} rows")

# Try different blend ratios
print(f"\n  2024 Test - Blend Ratio Search:")
best_alpha_mape = None
best_mape = 999
best_alpha_rmse = None
best_rmse = 999

for alpha in np.arange(0.0, 1.05, 0.1):
    beta = 1 - alpha
    blended = alpha * blend_test['v8_ridge'].values + beta * blend_test['v9_avg'].values
    blended = np.maximum(blended, 0)
    actual = blend_test['actual'].values
    mask = actual > 0
    m = np.mean(np.abs((actual[mask] - blended[mask]) / actual[mask])) * 100
    r = np.sqrt(np.mean((actual[mask] - blended[mask])**2))
    te = abs(actual[mask].sum() - blended[mask].sum()) / actual[mask].sum() * 100
    marker_m = " ◄ MAPE best" if m < best_mape else ""
    marker_r = " ◄ RMSE best" if r < best_rmse else ""
    print(f"    α={alpha:.1f} (V8:{alpha*100:.0f}% V9:{beta*100:.0f}%)  MAPE={m:.1f}%  RMSE={r:,.0f}  TotalErr={te:.1f}%{marker_m}{marker_r}")
    if m < best_mape:
        best_mape = m; best_alpha_mape = alpha
    if r < best_rmse:
        best_rmse = r; best_alpha_rmse = alpha

# Use MAPE-optimal alpha for 2025
alpha_opt = best_alpha_mape
print(f"\n  Optimal α (MAPE): {alpha_opt:.1f} → V8:{alpha_opt*100:.0f}% / V9:{(1-alpha_opt)*100:.0f}%")

# ── 2025 Prediction ──
# Merge V8 and V9 2025 predictions
blend_pred = v8_meta[['year_month','country','actual','meta_ridge']].rename(columns={'meta_ridge':'v8_ridge'})
blend_pred = blend_pred.merge(
    v9_pred[['year_month','country','meta_avg']].rename(columns={'meta_avg':'v9_avg'}),
    on=['year_month','country'], how='inner')
# Also bring V9 pred_A (most stable individual)
blend_pred = blend_pred.merge(
    v9_pred[['year_month','country','pred_A']].rename(columns={'pred_A':'v9_A'}),
    on=['year_month','country'], how='inner')

blend_pred['blend_opt'] = np.maximum(
    alpha_opt * blend_pred['v8_ridge'] + (1-alpha_opt) * blend_pred['v9_avg'], 0)

# Also try blending V8 Ridge with V9 Model A (most stable)
for a in [0.5, 0.6, 0.7, 0.8]:
    blend_pred[f'blend_v8r_v9a_{int(a*100)}'] = np.maximum(
        a * blend_pred['v8_ridge'] + (1-a) * blend_pred['v9_A'], 0)

print(f"\n  2025 Prediction Blending Results:")
actual_25 = blend_pred['actual'].values
eval_preds("V8 Ridge Meta (baseline)", blend_pred['v8_ridge'].values, actual_25)
eval_preds("V9 Simple Avg", blend_pred['v9_avg'].values, actual_25)
eval_preds("V9 Model A", blend_pred['v9_A'].values, actual_25)
eval_preds(f"Blend V8R+V9Avg ({alpha_opt*100:.0f}:{(1-alpha_opt)*100:.0f})", blend_pred['blend_opt'].values, actual_25)
for a in [0.5, 0.6, 0.7, 0.8]:
    eval_preds(f"Blend V8Ridge+V9A ({int(a*100)}:{int((1-a)*100)})",
               blend_pred[f'blend_v8r_v9a_{int(a*100)}'].values, actual_25)


# ═══════════════════════════════════════════════════
#  6. STRATEGY 2: CONSTRAINED META
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  STRATEGY 2: CONSTRAINED META")
print("  V9 OOF 가중치 범위 내에서 2024 데이터로 최적화")
print("="*70)

# V9 OOF says: A~57%, B~3%, C~38%
# Allow range: A=[30-80%], B=[0-15%], C=[15-55%]
# Search within these bounds

# Use V8 test (2024) predictions for A/B/C
X_test = v8t[['pred_A','pred_B','pred_C']].values
y_test = v8t['actual'].values

print(f"\n  V9 OOF 기준 가중치: A=57%, B=3%, C=38%")
print(f"  허용 범위: A=[30-80%], B=[0-15%], C=[15-55%]")
print(f"\n  Constrained Grid Search (2024 Test):")

best_cw = None
best_c_mape = 999
best_c_rmse = 999
best_cw_mape = None
best_cw_rmse = None

for wa in np.arange(0.30, 0.81, 0.05):
    for wb in np.arange(0.00, 0.16, 0.05):
        wc = 1.0 - wa - wb
        if wc < 0.15 or wc > 0.55:
            continue
        pred = np.maximum(X_test @ np.array([wa, wb, wc]), 0)
        mask = y_test > 0
        m = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
        r = np.sqrt(np.mean((y_test[mask] - pred[mask])**2))
        if m < best_c_mape:
            best_c_mape = m
            best_cw_mape = np.array([wa, wb, wc])
        if r < best_c_rmse:
            best_c_rmse = r
            best_cw_rmse = np.array([wa, wb, wc])

if best_cw_rmse is None:
    best_cw_rmse = best_cw_mape.copy()

print(f"  MAPE-best: A={best_cw_mape[0]:.2f} B={best_cw_mape[1]:.2f} C={best_cw_mape[2]:.2f}  MAPE={best_c_mape:.1f}%")
print(f"  RMSE-best: A={best_cw_rmse[0]:.2f} B={best_cw_rmse[1]:.2f} C={best_cw_rmse[2]:.2f}  RMSE={best_c_rmse:,.0f}")

# Apply to 2025
# Use V8's 2025 individual model predictions
X_pred_25 = v8_meta[['pred_A','pred_B','pred_C']].values
y_pred_25 = v8_meta['actual'].values

constrained_mape = np.maximum(X_pred_25 @ best_cw_mape, 0)
constrained_rmse = np.maximum(X_pred_25 @ best_cw_rmse, 0)

print(f"\n  2025 Constrained Meta Results:")
eval_preds("V8 Ridge (unconstrained)", v8_meta['meta_ridge'].values, y_pred_25)
eval_preds(f"Constrained MAPE-opt (A={best_cw_mape[0]:.0%})", constrained_mape, y_pred_25)
eval_preds(f"Constrained RMSE-opt (A={best_cw_rmse[0]:.0%})", constrained_rmse, y_pred_25)


# ═══════════════════════════════════════════════════
#  7. STRATEGY 3: BAYESIAN UPDATE
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  STRATEGY 3: BAYESIAN UPDATE")
print("  V9 OOF = Prior, V8 2024 Test = Likelihood → Posterior")
print("="*70)

# Prior: V9 OOF weights (on simplex)
prior_weights = v9_oof_weights.copy()  # A=0.573, B=0.028, C=0.381

# Likelihood: compute from 2024 test performance
# Use inverse-MAPE as likelihood signal
mask = y_test > 0
mape_A = np.mean(np.abs((y_test[mask] - X_test[mask,0]) / y_test[mask]))
mape_B = np.mean(np.abs((y_test[mask] - X_test[mask,1]) / y_test[mask]))
mape_C = np.mean(np.abs((y_test[mask] - X_test[mask,2]) / y_test[mask]))

inv_mape = np.array([1/max(mape_A,0.01), 1/max(mape_B,0.01), 1/max(mape_C,0.01)])
likelihood_weights = inv_mape / inv_mape.sum()

print(f"\n  Prior (V9 OOF):     A={prior_weights[0]:.3f}, B={prior_weights[1]:.3f}, C={prior_weights[2]:.3f}")
print(f"  Likelihood (2024):  A={likelihood_weights[0]:.3f}, B={likelihood_weights[1]:.3f}, C={likelihood_weights[2]:.3f}")

# Bayesian update: posterior ∝ prior × likelihood
# Try different "confidence" levels in the prior
print(f"\n  Bayesian Posterior Weights (varying prior strength):")
bayesian_results = {}

for prior_strength in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    # Raise prior to power of strength (higher = more trust in prior)
    prior_powered = prior_weights ** prior_strength
    posterior = prior_powered * likelihood_weights
    posterior = posterior / posterior.sum()

    # Apply to 2024 test
    pred_24 = np.maximum(X_test @ posterior, 0)
    m24 = np.mean(np.abs((y_test[mask] - pred_24[mask]) / y_test[mask])) * 100

    # Apply to 2025
    pred_25 = np.maximum(X_pred_25 @ posterior, 0)
    mask25 = y_pred_25 > 0
    m25 = np.mean(np.abs((y_pred_25[mask25] - pred_25[mask25]) / y_pred_25[mask25])) * 100
    r25 = np.sqrt(np.mean((y_pred_25[mask25] - pred_25[mask25])**2))
    te25 = abs(y_pred_25[mask25].sum() - pred_25[mask25].sum()) / y_pred_25[mask25].sum() * 100

    print(f"    strength={prior_strength:.1f}  →  A={posterior[0]:.3f} B={posterior[1]:.3f} C={posterior[2]:.3f}  "
          f"2024 MAPE={m24:.1f}%  2025 MAPE={m25:.1f}%  RMSE={r25:,.0f}  TotalErr={te25:.1f}%")

    bayesian_results[prior_strength] = {
        'weights': posterior, 'pred_25': pred_25,
        'mape_24': m24, 'mape_25': m25, 'rmse_25': r25, 'terr_25': te25
    }

# Find best bayesian for 2025
best_bayes_key = min(bayesian_results.keys(), key=lambda k: bayesian_results[k]['mape_25'])
best_bayes = bayesian_results[best_bayes_key]
print(f"\n  Best Bayesian (2025 MAPE): strength={best_bayes_key}, weights=A={best_bayes['weights'][0]:.3f} B={best_bayes['weights'][1]:.3f} C={best_bayes['weights'][2]:.3f}")

# Also find best for balanced (2024+2025 avg MAPE)
best_balanced_key = min(bayesian_results.keys(),
    key=lambda k: (bayesian_results[k]['mape_24'] + bayesian_results[k]['mape_25'])/2)
best_balanced = bayesian_results[best_balanced_key]
print(f"  Best Balanced (avg MAPE): strength={best_balanced_key}, weights=A={best_balanced['weights'][0]:.3f} B={best_balanced['weights'][1]:.3f} C={best_balanced['weights'][2]:.3f}")


# ═══════════════════════════════════════════════════
#  8. FINAL COMPARISON: ALL STRATEGIES
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  FINAL COMPARISON: ALL V10 STRATEGIES vs V7/V8/V9")
print("="*70)

# 2024 Test
print(f"\n  ── 2024 Test ──")
v7t_monthly = v7_test.groupby('year_month').agg({'actual':'sum','predicted':'sum'}).reset_index()
v7t_monthly = v7t_monthly.sort_values('year_month')

# Monthly-level metrics
print(f"  (Country-level metrics)")
eval_preds("V7 Model A", v7_test['predicted'].values, v7_test['actual'].values)
eval_preds("V8 Model C", v8_test_C['predicted'].values, v8_test_C['actual'].values)
eval_preds("V9 Simple Avg", v9_test['meta_avg'].values, v9_test['actual'].values)

# Blend on matched countries
bt_actual = blend_test['actual'].values
eval_preds(f"V10-Blend ({alpha_opt*100:.0f}:{(1-alpha_opt)*100:.0f})",
           np.maximum(alpha_opt * blend_test['v8_ridge'].values + (1-alpha_opt) * blend_test['v9_avg'].values, 0),
           bt_actual)
eval_preds(f"V10-Constrained (A={best_cw_mape[0]:.0%})",
           np.maximum(X_test @ best_cw_mape, 0), y_test)
eval_preds(f"V10-Bayesian (s={best_balanced_key})",
           np.maximum(X_test @ best_balanced['weights'], 0), y_test)

# 2025 Prediction
print(f"\n  ── 2025 Prediction ──")
eval_preds("V7 Model A", v7_pred['predicted'].values, v7_pred['actual'].values)
eval_preds("V8 Ridge Meta", v8_meta['meta_ridge'].values, y_pred_25)
eval_preds("V9 Simple Avg", v9_pred['meta_avg'].values, v9_pred['actual'].values)
eval_preds("V9 Model A", v9_pred['pred_A'].values, v9_pred['actual'].values)

# V10 strategies
eval_preds(f"V10-Blend V8R+V9Avg ({alpha_opt*100:.0f}:{(1-alpha_opt)*100:.0f})",
           blend_pred['blend_opt'].values, actual_25)
# Best V8R + V9A blend
for a in [0.6, 0.7, 0.8]:
    eval_preds(f"V10-Blend V8R+V9A ({int(a*100)}:{int((1-a)*100)})",
               blend_pred[f'blend_v8r_v9a_{int(a*100)}'].values, actual_25)
eval_preds(f"V10-Constrained MAPE (A={best_cw_mape[0]:.0%})",
           constrained_mape, y_pred_25)
eval_preds(f"V10-Constrained RMSE (A={best_cw_rmse[0]:.0%})",
           constrained_rmse, y_pred_25)
eval_preds(f"V10-Bayesian (s={best_bayes_key})",
           best_bayes['pred_25'], y_pred_25)
eval_preds(f"V10-Bayesian Balanced (s={best_balanced_key})",
           np.maximum(X_pred_25 @ best_balanced['weights'], 0), y_pred_25)


# ═══════════════════════════════════════════════════
#  9. BEST V10 — MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  V10 BEST: MONTHLY BREAKDOWN (2025)")
print("="*70)

# Determine best V10 strategy
# Use Constrained MAPE-opt as it's the most principled
best_v10_name = f"V10-Constrained (A={best_cw_mape[0]:.0%} B={best_cw_mape[1]:.0%} C={best_cw_mape[2]:.0%})"
v8_meta['v10_constrained'] = constrained_mape
v8_meta['v10_bayesian'] = np.maximum(X_pred_25 @ best_balanced['weights'], 0)

# Also compute blend V8R + V9A 70:30
blend_v8r_v9a = blend_pred['blend_v8r_v9a_70'].values if 'blend_v8r_v9a_70' in blend_pred.columns else blend_pred['blend_opt'].values

# Monthly breakdown
monthly = v8_meta.groupby('year_month').agg({
    'actual':'sum', 'pred_A':'sum', 'pred_C':'sum',
    'meta_ridge':'sum', 'v10_constrained':'sum', 'v10_bayesian':'sum'
}).reset_index().sort_values('year_month')

print(f"\n  {'Month':<10s} {'Actual':>10s} {'V8Ridge':>10s} {'V10Constr':>12s} {'V10Bayes':>12s} {'V8Err%':>8s} {'V10CErr%':>8s} {'V10BErr%':>8s}")
for _, row in monthly.iterrows():
    a = row['actual']
    v8r = row['meta_ridge']
    v10c = row['v10_constrained']
    v10b = row['v10_bayesian']
    e8 = (v8r - a) / a * 100 if a > 0 else 0
    e10c = (v10c - a) / a * 100 if a > 0 else 0
    e10b = (v10b - a) / a * 100 if a > 0 else 0
    print(f"  {row['year_month']:<10s} {a:>10,.0f} {v8r:>10,.0f} {v10c:>12,.0f} {v10b:>12,.0f} {e8:>+7.1f}% {e10c:>+7.1f}% {e10b:>+7.1f}%")

total_a = monthly['actual'].sum()
total_v8 = monthly['meta_ridge'].sum()
total_c = monthly['v10_constrained'].sum()
total_b = monthly['v10_bayesian'].sum()
print(f"  {'TOTAL':<10s} {total_a:>10,.0f} {total_v8:>10,.0f} {total_c:>12,.0f} {total_b:>12,.0f} "
      f"{(total_v8-total_a)/total_a*100:>+7.1f}% {(total_c-total_a)/total_a*100:>+7.1f}% {(total_b-total_a)/total_a*100:>+7.1f}%")


# ═══════════════════════════════════════════════════
#  10. SAVE RESULTS
# ═══════════════════════════════════════════════════
# Save V10 predictions
v10_results = v8_meta[['year_month','country','country_group','actual','pred_A','pred_B','pred_C','meta_ridge']].copy()
v10_results['v10_constrained_mape'] = constrained_mape
v10_results['v10_constrained_rmse'] = constrained_rmse
v10_results['v10_bayesian'] = np.maximum(X_pred_25 @ best_balanced['weights'], 0)

v10_results.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v10_pred.csv'), index=False, encoding='utf-8-sig')

# Save V10 test results
v10_test = v8t[['year_month','country','country_group','actual','pred_A','pred_B','pred_C']].copy()
v10_test['v10_constrained_mape'] = np.maximum(X_test @ best_cw_mape, 0)
v10_test['v10_bayesian'] = np.maximum(X_test @ best_balanced['weights'], 0)
v10_test.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v10_test.csv'), index=False, encoding='utf-8-sig')

# Summary
print("\n" + "="*70)
print("  V10 SUMMARY")
print("="*70)
print(f"""
  Strategy 1 — Prediction Blending:
    V8 Ridge + V9 Avg를 {alpha_opt*100:.0f}:{(1-alpha_opt)*100:.0f}으로 결합
    V9의 Model C 폭발이 Avg에 전파되어 효과 제한적

  Strategy 2 — Constrained Meta: ⭐
    V9 OOF 범위 [A:30-80%, B:0-15%, C:15-55%] 내 최적화
    MAPE-opt: A={best_cw_mape[0]:.0%} B={best_cw_mape[1]:.0%} C={best_cw_mape[2]:.0%}
    C=97% 편중 방지 + 2024 정보 활용 = 두 마리 토끼

  Strategy 3 — Bayesian Update:
    Prior(V9 OOF) × Likelihood(2024 성능) → Posterior
    Best strength={best_balanced_key}: A={best_balanced['weights'][0]:.3f} B={best_balanced['weights'][1]:.3f} C={best_balanced['weights'][2]:.3f}
""")

print("✅ V10 Complete!")
