"""
V9: Information Diversity + Rolling Window OOF Meta-Stacking
=============================================================
V8: 정보 다양성 → error corr 0.99→0.61 (성공)
V9: + Rolling Window OOF → meta learner 정보 누출 제거

Rolling Window Strategy (COVID 2020-01~2023-06 제외):
  Fold 1: Train 2018       → OOF Predict 2019       (12개월)
  Fold 2: Train 2018~2019  → OOF Predict 2023H2     (6개월, COVID 후 첫 정상)
  Fold 3: Train 2018~2019 + 2023H2 → OOF Predict 2024 (12개월)
  Meta:   Fold 1~3 OOF로 가중치 학습 (30개월 × 25국)
  Final:  전체 non-COVID 학습 → 2025 예측
"""
import sys, os, unicodedata
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
import numpy as np, pandas as pd
from loader import V4DataLoader, COUNTRY_GROUPS, _read_csv_safe, _nfc, _normalize_yearmonth
from engineer import V4FeatureEngineer
from models import (GradientBoostingV4, RidgeRegression, WeightedLagModel,
                        StackedEnsemble, rmse, mape)
np.random.seed(42)

# ═══════════════════════════════════════════════════
#  1. DATA LOADING (same as V7/V8)
# ═══════════════════════════════════════════════════
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'country')
bp = DATA_DIR
print(f"Base: {bp}")

loader = V4DataLoader(bp); panel = loader.build_panel()

# ETC country
gdir = loader.global_dir; gc = None
for e in os.scandir(gdir):
    if '외래관광객 추이' in _nfc(e.name) and e.name.endswith('.csv'):
        gc = e.path; break
gdf = _read_csv_safe(gc); gdf.columns = [_nfc(c) for c in gdf.columns]
gdf['year_month'] = gdf['기준년월'].astype(str).apply(_normalize_yearmonth)
gdf['total_visitors'] = pd.to_numeric(gdf['방한 외래관광객'].astype(str).str.replace(',',''), errors='coerce')
gdf['exchange_rate'] = pd.to_numeric(gdf['환율(원)'].astype(str).str.replace(',',''), errors='coerce')
gdf['oil_price'] = pd.to_numeric(gdf['국제유가(달러)'].astype(str).str.replace(',',''), errors='coerce')
s24 = panel.groupby('year_month')['visitors'].sum().reset_index().rename(columns={'visitors':'s24'})
edf = gdf[['year_month','total_visitors','exchange_rate','oil_price']].merge(s24, on='year_month', how='left')
edf['visitors'] = edf['total_visitors'] - edf['s24'].fillna(0)
edf['country'] = 'ETC'; edf['country_group'] = 'ETC_group'
edf['year'] = edf['year_month'].str[:4].astype(int)
edf['month'] = edf['year_month'].str[5:7].astype(int)
edf = edf[['year_month','country','country_group','visitors','exchange_rate','oil_price','year','month']]
pe = pd.concat([panel, edf], ignore_index=True).sort_values(['country','year_month']).reset_index(drop=True)

# Feature engineering
eng = V4FeatureEngineer(include_tier2=True)
df_full = eng.build_features(pe)
all_feats = eng.get_feature_columns(df_full, tier=2)
print(f"Panel: {pe.shape[0]} rows, {pe['country'].nunique()} countries, Features: {len(all_feats)}")

# ═══════════════════════════════════════════════════
#  2. IMPUTATION + FEATURE SETS (same as V8)
# ═══════════════════════════════════════════════════
MS = {'멕시코','사우디아라비야','아랍에미리트','카자흐스탄','튀르키예','ETC'}
MH = {'몽골','사우디아라비야','싱가포르','필리핀','홍콩','ETC'}
SF = ['revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
      'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag','spend_x_fx']
HF = ['hallyu_index','hallyu_x_interest']
IF_cols = ['image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
      'image_kdrama_pct','culture_affinity','culture_x_interest']

def impute_groupmedian(d):
    d = d.copy()
    for col in SF + HF + IF_cols:
        if col not in d.columns: continue
        ms = MS if col in SF else MH
        for c in ms:
            m = d['country'] == c
            if not m.any(): continue
            g = d.loc[m, 'country_group'].iloc[0]
            gm = (d['country_group'] == g) & (~d['country'].isin(ms))
            d.loc[m, col] = d.loc[m, col].fillna(d.loc[gm, col].median())
    return d

df_imp = impute_groupmedian(df_full)

# Feature sets (V8 information diversity)
FEATS_A = [f for f in [
    'visitor_lag_1','visitor_lag_2','visitor_lag_3','visitor_lag_6','visitor_lag_12',
    'visitor_ma_3','visitor_ma_6','visitor_ma_12','visitor_yoy_growth','recovery_ratio',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','month_sin','month_cos',
] if f in df_imp.columns]

FEATS_B = [f for f in [
    'exchange_rate','exchange_rate_lag_1','exchange_rate_ma_3',
    'exchange_rate_change_pct','exchange_rate_volatility',
    'oil_price','oil_price_lag_1','oil_price_ma_3',
    'travel_interest_pct','travel_interest_lag_1','travel_interest_lag_2','travel_interest_lag_3',
    'global_travel_interest_pct','global_interest_lag_1','global_interest_lag_2','global_interest_lag_3',
    'spend_x_fx','covid_period','post_covid',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','visitor_lag_12',
] if f in df_imp.columns]

FEATS_C = [f for f in [
    'month_sin','month_cos','quarter','is_peak_season','is_summer','is_winter',
    'weather_comfort','temp_lag_1','precip_lag_1','temp_anomaly','heavy_rain',
    'event_national_lag_1','event_seoul_lag_1','event_seoul_ratio','event_intensity','event_weighted',
    'hallyu_index','hallyu_x_interest',
    'image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
    'image_kdrama_pct','culture_affinity','culture_x_interest',
    'revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
    'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag',
    'tourism_dominant','business_ratio','covid_period','post_covid',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','visitor_lag_12',
] if f in df_imp.columns]

print(f"Feature sets: A={len(FEATS_A)}, B={len(FEATS_B)}, C={len(FEATS_C)}")

# ═══════════════════════════════════════════════════
#  3. ROLLING WINDOW OOF FUNCTION
# ═══════════════════════════════════════════════════
def train_predict_fold(df, feats, train_mask, pred_mask, label):
    """Train on train_mask rows, predict pred_mask rows. Returns predictions per group."""
    train = df[train_mask].copy()
    pred_set = df[pred_mask].copy()

    all_preds = []
    for group in sorted(train['country_group'].unique()):
        gt = train[train['country_group'] == group]
        gp = pred_set[pred_set['country_group'] == group]
        if len(gp) == 0: continue

        vf = [c for c in feats if c in gt.columns]
        Xt = gt[vf].values.astype(float); yt = gt['visitors'].values.astype(float)
        Xp = gp[vf].values.astype(float); yp = gp['visitors'].values.astype(float)
        meta = gp[['year_month','country','country_group']].copy()

        ok = ~np.isnan(yt) & (yt > 0); Xt, yt = Xt[ok], yt[ok]
        Xt = np.nan_to_num(Xt, nan=0.0)
        Xp = np.nan_to_num(Xp, nan=0.0)

        if len(Xt) < 5: continue

        # 3 base models
        models = {
            'GBM': GradientBoostingV4(n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=5),
            'Ridge': RidgeRegression(alpha=10.0),
            'WLag': WeightedLagModel(),
        }
        fitted = {}
        for n, m in models.items():
            try:
                m.fit(Xt, yt, feature_names=vf); fitted[n] = m
            except: pass

        if len(fitted) < 2: continue

        # StackedEnsemble — but need validation data for weights
        # Use last 20% of training as validation for weight fitting
        split_idx = max(int(len(Xt) * 0.8), len(Xt) - 12)
        Xv_w = Xt[split_idx:]; yv_w = yt[split_idx:]
        if len(Xv_w) > 3:
            st = StackedEnsemble(list(fitted.values()), list(fitted.keys()))
            st.fit_weights(Xv_w, yv_w)
            fitted['Stacked'] = st

        # Predict with each fitted model → store for meta
        ok_pred = ~np.isnan(yp) & (yp > 0)
        meta_valid = meta.iloc[ok_pred.nonzero()[0]].copy()
        Xp_valid = Xp[ok_pred]
        yp_valid = yp[ok_pred]

        if len(Xp_valid) == 0: continue

        # Use best single model by internal validation
        best_n = 'Stacked' if 'Stacked' in fitted else list(fitted.keys())[0]
        pred_vals = np.maximum(fitted[best_n].predict(Xp_valid), 0)
        # Sanity clip: predictions shouldn't exceed 5x max training target per group
        max_train = yt.max()
        pred_vals = np.minimum(pred_vals, max_train * 5)

        meta_valid = meta_valid.copy()
        meta_valid['actual'] = yp_valid
        meta_valid['predicted'] = pred_vals
        all_preds.append(meta_valid)

    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════
#  4. ROLLING WINDOW FOLDS
# ═══════════════════════════════════════════════════
COVID_S, COVID_E = '2020-01', '2023-06'

def is_covid(ym):
    return ym >= COVID_S and ym <= COVID_E

# Fold definitions
# NOTE: Skip 2023H2 as prediction target — lag features reference COVID months
#       (visitor_lag_6 for 2023-07 = 2023-01 = COVID, near-zero → model breaks)
# Use clean non-COVID periods only for both train AND predict
folds = [
    {
        'name': 'Fold 1: Train 2018 → Predict 2019H1',
        'train': lambda ym: '2018-01' <= ym <= '2018-12',
        'pred':  lambda ym: '2019-01' <= ym <= '2019-06',
    },
    {
        'name': 'Fold 2: Train 2018~2019H1 → Predict 2019H2',
        'train': lambda ym: '2018-01' <= ym <= '2019-06',
        'pred':  lambda ym: '2019-07' <= ym <= '2019-12',
    },
    {
        'name': 'Fold 3: Train 2018~2019 → Predict 2024H1',
        'train': lambda ym: ('2018-01' <= ym <= '2019-12') or ('2023-07' <= ym <= '2023-12'),
        'pred':  lambda ym: '2024-01' <= ym <= '2024-06',
    },
    {
        'name': 'Fold 4: Train 2018~2019+2023H2+2024H1 → Predict 2024H2',
        'train': lambda ym: ('2018-01' <= ym <= '2019-12') or ('2023-07' <= ym <= '2024-06'),
        'pred':  lambda ym: '2024-07' <= ym <= '2024-12',
    },
]

print("\n" + "="*70)
print("  V9: ROLLING WINDOW OOF PREDICTIONS")
print("="*70)

oof_A, oof_B, oof_C = [], [], []

for fold in folds:
    train_mask = df_imp['year_month'].apply(fold['train']).values
    pred_mask = df_imp['year_month'].apply(fold['pred']).values

    print(f"\n--- {fold['name']} ---")
    print(f"  Train: {train_mask.sum()} rows, Pred: {pred_mask.sum()} rows")

    pA = train_predict_fold(df_imp, FEATS_A, train_mask, pred_mask, 'A')
    pB = train_predict_fold(df_imp, FEATS_B, train_mask, pred_mask, 'B')
    pC = train_predict_fold(df_imp, FEATS_C, train_mask, pred_mask, 'C')

    if not pA.empty: oof_A.append(pA)
    if not pB.empty: oof_B.append(pB)
    if not pC.empty: oof_C.append(pC)

    # Quick fold performance
    for label, p in [('A', pA), ('B', pB), ('C', pC)]:
        if not p.empty:
            m = np.mean(np.abs((p['actual'] - p['predicted']) / p['actual'])) * 100
            print(f"  Model {label}: {len(p)} predictions, MAPE={m:.1f}%")

oof_A = pd.concat(oof_A, ignore_index=True) if oof_A else pd.DataFrame()
oof_B = pd.concat(oof_B, ignore_index=True) if oof_B else pd.DataFrame()
oof_C = pd.concat(oof_C, ignore_index=True) if oof_C else pd.DataFrame()

print(f"\nTotal OOF: A={len(oof_A)}, B={len(oof_B)}, C={len(oof_C)}")

# ═══════════════════════════════════════════════════
#  5. OOF CORRELATION CHECK
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  OOF PREDICTION CORRELATION")
print("="*70)

oof_merged = oof_A[['year_month','country','actual','predicted']].rename(columns={'predicted':'pred_A'})
oof_merged = oof_merged.merge(
    oof_B[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),
    on=['year_month','country'], how='inner')
oof_merged = oof_merged.merge(
    oof_C[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),
    on=['year_month','country'], how='inner')

print(f"  OOF merged: {len(oof_merged)} rows")

if len(oof_merged) > 10:
    corr_AB = np.corrcoef(oof_merged['pred_A'], oof_merged['pred_B'])[0,1]
    corr_AC = np.corrcoef(oof_merged['pred_A'], oof_merged['pred_C'])[0,1]
    corr_BC = np.corrcoef(oof_merged['pred_B'], oof_merged['pred_C'])[0,1]

    oof_merged['err_A'] = oof_merged['pred_A'] - oof_merged['actual']
    oof_merged['err_B'] = oof_merged['pred_B'] - oof_merged['actual']
    oof_merged['err_C'] = oof_merged['pred_C'] - oof_merged['actual']

    ecorr_AB = np.corrcoef(oof_merged['err_A'], oof_merged['err_B'])[0,1]
    ecorr_AC = np.corrcoef(oof_merged['err_A'], oof_merged['err_C'])[0,1]
    ecorr_BC = np.corrcoef(oof_merged['err_B'], oof_merged['err_C'])[0,1]

    print(f"\n  Prediction Correlations (OOF):")
    print(f"    A↔B: {corr_AB:.4f}")
    print(f"    A↔C: {corr_AC:.4f}")
    print(f"    B↔C: {corr_BC:.4f}")
    print(f"\n  Error Correlations (OOF):")
    print(f"    A↔B: {ecorr_AB:.4f}")
    print(f"    A↔C: {ecorr_AC:.4f}")
    print(f"    B↔C: {ecorr_BC:.4f}")
    print(f"\n  Avg |Pred Corr|: {(abs(corr_AB)+abs(corr_AC)+abs(corr_BC))/3:.4f}")
    print(f"  Avg |Error Corr|: {(abs(ecorr_AB)+abs(ecorr_AC)+abs(ecorr_BC))/3:.4f}")


# ═══════════════════════════════════════════════════
#  6. META LEARNER ON OOF
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  META-LEARNER TRAINING ON OOF")
print("="*70)

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

X_oof = oof_merged[['pred_A','pred_B','pred_C']].values
y_oof = oof_merged['actual'].values

# Ridge Meta on OOF
ridge_oof = SimpleRidge(alpha=1.0)
ridge_oof.fit(X_oof, y_oof)
oof_ridge = np.maximum(ridge_oof.predict(X_oof), 0)

# Simple average
oof_avg = np.maximum((oof_merged['pred_A'] + oof_merged['pred_B'] + oof_merged['pred_C']).values / 3, 0)

# Inverse-MAPE weighted
mape_a = np.mean(np.abs((y_oof - oof_merged['pred_A'].values) / y_oof))
mape_b = np.mean(np.abs((y_oof - oof_merged['pred_B'].values) / y_oof))
mape_c = np.mean(np.abs((y_oof - oof_merged['pred_C'].values) / y_oof))
inv_m = np.array([1/max(mape_a,0.01), 1/max(mape_b,0.01), 1/max(mape_c,0.01)])
w_imape = inv_m / inv_m.sum()
oof_wmean = np.maximum(
    oof_merged['pred_A'].values * w_imape[0] +
    oof_merged['pred_B'].values * w_imape[1] +
    oof_merged['pred_C'].values * w_imape[2], 0)

print(f"  Ridge OOF coef: A={ridge_oof.coef_[0]:.4f}, B={ridge_oof.coef_[1]:.4f}, C={ridge_oof.coef_[2]:.4f}")
print(f"  Ridge OOF intercept: {ridge_oof.intercept_:.2f}")
print(f"  IMAPE weights: A={w_imape[0]:.4f}, B={w_imape[1]:.4f}, C={w_imape[2]:.4f}")

# OOF performance
def eval_preds(name, pred, actual):
    r = np.sqrt(np.mean((actual - pred)**2))
    m = np.mean(np.abs((actual - pred) / actual)) * 100
    te = abs(actual.sum() - pred.sum()) / actual.sum() * 100
    print(f"  {name:<25s}  RMSE={r:>10,.0f}  MAPE={m:>6.1f}%  TotalErr={te:>5.1f}%")
    return {'name':name, 'rmse':r, 'mape':m, 'terr':te}

print(f"\n  OOF Performance (2019 + 2023H2 + 2024 combined):")
eval_preds("Model A (Lag)", oof_merged['pred_A'].values, y_oof)
eval_preds("Model B (Macro)", oof_merged['pred_B'].values, y_oof)
eval_preds("Model C (Event)", oof_merged['pred_C'].values, y_oof)
eval_preds("Simple Average", oof_avg, y_oof)
eval_preds("Weighted Average", oof_wmean, y_oof)
eval_preds("Ridge Meta (OOF)", oof_ridge, y_oof)


# ═══════════════════════════════════════════════════
#  7. FINAL: TRAIN ALL → PREDICT 2025
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  FINAL: FULL TRAINING → 2025 PREDICTION")
print("="*70)

# Train on all non-COVID data up to 2024
final_train_mask = df_imp['year_month'].apply(
    lambda ym: ym <= '2024-12' and not is_covid(ym) and ym >= '2018-01'
).values
final_pred_mask = df_imp['year_month'].apply(
    lambda ym: '2025-01' <= ym <= '2025-12'
).values

# Also get 2024 test predictions from final model (for direct comparison with V7/V8)
test_pred_mask = df_imp['year_month'].apply(
    lambda ym: '2024-01' <= ym <= '2024-12'
).values
# For 2024 test, train on everything before 2024
test_train_mask = df_imp['year_month'].apply(
    lambda ym: ym <= '2023-12' and not is_covid(ym) and ym >= '2018-01'
).values

print(f"  Test train: {test_train_mask.sum()} rows")
print(f"  Test pred:  {test_pred_mask.sum()} rows")
print(f"  Final train: {final_train_mask.sum()} rows")
print(f"  Final pred:  {final_pred_mask.sum()} rows")

# 2024 test predictions (for comparison)
print("\n--- 2024 Test Predictions ---")
testA = train_predict_fold(df_imp, FEATS_A, test_train_mask, test_pred_mask, 'A-test')
testB = train_predict_fold(df_imp, FEATS_B, test_train_mask, test_pred_mask, 'B-test')
testC = train_predict_fold(df_imp, FEATS_C, test_train_mask, test_pred_mask, 'C-test')

for label, t in [('A', testA), ('B', testB), ('C', testC)]:
    if not t.empty:
        m = np.mean(np.abs((t['actual'] - t['predicted']) / t['actual'])) * 100
        print(f"  Model {label}: MAPE={m:.1f}%")

# 2025 predictions
print("\n--- 2025 Predictions ---")
predA = train_predict_fold(df_imp, FEATS_A, final_train_mask, final_pred_mask, 'A-final')
predB = train_predict_fold(df_imp, FEATS_B, final_train_mask, final_pred_mask, 'B-final')
predC = train_predict_fold(df_imp, FEATS_C, final_train_mask, final_pred_mask, 'C-final')

for label, p in [('A', predA), ('B', predB), ('C', predC)]:
    if not p.empty:
        has_actual = p['actual'].notna() & (p['actual'] > 0)
        if has_actual.sum() > 0:
            m = np.mean(np.abs((p.loc[has_actual,'actual'] - p.loc[has_actual,'predicted']) / p.loc[has_actual,'actual'])) * 100
            print(f"  Model {label}: {len(p)} predictions, MAPE={m:.1f}%")

# ═══════════════════════════════════════════════════
#  8. APPLY META LEARNER TO 2024 TEST & 2025
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  META-STACKING RESULTS")
print("="*70)

def apply_meta(rA, rB, rC, ridge_model, weights, label):
    """Merge 3 model predictions and apply meta learners"""
    merged = rA[['year_month','country','country_group','actual','predicted']].rename(columns={'predicted':'pred_A'})
    merged = merged.merge(rB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),
                          on=['year_month','country'], how='inner')
    merged = merged.merge(rC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),
                          on=['year_month','country'], how='inner')

    Xm = merged[['pred_A','pred_B','pred_C']].values
    merged['meta_ridge'] = np.maximum(ridge_model.predict(Xm), 0)
    merged['meta_avg'] = np.maximum((merged['pred_A'] + merged['pred_B'] + merged['pred_C']) / 3, 0)
    merged['meta_wmean'] = np.maximum(
        merged['pred_A'] * weights[0] + merged['pred_B'] * weights[1] + merged['pred_C'] * weights[2], 0)

    print(f"\n  {label} ({len(merged)} rows):")
    has_actual = merged['actual'].notna() & (merged['actual'] > 0)
    if has_actual.sum() > 0:
        mv = merged[has_actual]
        actual = mv['actual'].values
        for name, col in [('Model A','pred_A'),('Model B','pred_B'),('Model C','pred_C'),
                          ('Simple Avg','meta_avg'),('Weighted Avg','meta_wmean'),('Ridge Meta (OOF)','meta_ridge')]:
            pred = mv[col].values
            r = np.sqrt(np.mean((actual - pred)**2))
            m = np.mean(np.abs((actual - pred) / actual)) * 100
            te = abs(actual.sum() - pred.sum()) / actual.sum() * 100
            print(f"    {name:<25s}  RMSE={r:>10,.0f}  MAPE={m:>6.1f}%  TotalErr={te:>5.1f}%")

    return merged

# 2024 Test with OOF meta
test_meta = apply_meta(testA, testB, testC, ridge_oof, w_imape, "2024 TEST")

# 2025 Predictions with OOF meta
pred_meta = apply_meta(predA, predB, predC, ridge_oof, w_imape, "2025 PREDICTIONS")

# ═══════════════════════════════════════════════════
#  9. MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  2025 MONTHLY BREAKDOWN")
print("="*70)

if not pred_meta.empty:
    monthly = pred_meta.groupby('year_month').agg({
        'pred_A':'sum','pred_B':'sum','pred_C':'sum',
        'meta_ridge':'sum','meta_avg':'sum','meta_wmean':'sum','actual':'sum'
    }).reset_index().sort_values('year_month')

    print(f"\n  {'Month':<10s} {'Actual':>10s} {'Model A':>10s} {'Model C':>10s} {'Ridge(OOF)':>12s} {'Err%':>7s}")
    for _, row in monthly.iterrows():
        a = row['actual']
        r = row['meta_ridge']
        err = (r - a) / a * 100 if a > 0 else 0
        print(f"  {row['year_month']:<10s} {a:>10,.0f} {row['pred_A']:>10,.0f} {row['pred_C']:>10,.0f} {r:>12,.0f} {err:>+6.1f}%")

    total_a = monthly['actual'].sum()
    total_r = monthly['meta_ridge'].sum()
    total_err = (total_r - total_a) / total_a * 100
    print(f"  {'TOTAL':<10s} {total_a:>10,.0f} {monthly['pred_A'].sum():>10,.0f} {monthly['pred_C'].sum():>10,.0f} {total_r:>12,.0f} {total_err:>+6.1f}%")

    # Per-month absolute error average
    monthly['abs_err_ridge'] = np.abs((monthly['meta_ridge'] - monthly['actual']) / monthly['actual'] * 100)
    monthly['abs_err_A'] = np.abs((monthly['pred_A'] - monthly['actual']) / monthly['actual'] * 100)
    monthly['abs_err_C'] = np.abs((monthly['pred_C'] - monthly['actual']) / monthly['actual'] * 100)
    print(f"\n  Avg Monthly |Error|:")
    print(f"    Model A:       {monthly['abs_err_A'].mean():.1f}%")
    print(f"    Model C:       {monthly['abs_err_C'].mean():.1f}%")
    print(f"    Ridge(OOF):    {monthly['abs_err_ridge'].mean():.1f}%")

# Save
test_meta.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v9_test_meta.csv'), index=False, encoding='utf-8-sig')
pred_meta.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v9_pred_meta.csv'), index=False, encoding='utf-8-sig')

print("\n✅ V9 Complete!")
