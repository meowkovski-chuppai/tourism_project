"""
V8: Information Diversity Stacking
===================================
Professor's key insight: "정보 다양성"이 핵심이지 "전처리 다양성"이 아니다.

V5~V7: 같은 feature + missing value 전략만 다름 → pred corr 0.999 → stacking 무의미
V8: 완전히 다른 feature set → error pattern 다양성 → stacking 효과 기대

Model A (Lag/History):    과거 방문 패턴 중심
Model B (Macro/Economic): 환율, 유가, 여행관심도 중심
Model C (Event/Season):   계절성, 날씨, 축제, 한류, 이미지, 만족도 중심

Train: 2018-01 ~ 2023-12 (COVID 2020-01~2023-06 제외)
Test:  2024-01 ~ 2024-12
Predict: 2025-01 ~ 2025-12
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

TRAIN_START = '2018-01'
COVID_S = '2020-01'
COVID_E = '2023-06'

# ═══════════════════════════════════════════════════
#  1. DATA LOADING (same as V7)
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
print(f"Panel: {pe.shape[0]} rows, {pe['country'].nunique()} countries")

# Feature engineering (full)
eng = V4FeatureEngineer(include_tier2=True)
df_full = eng.build_features(pe)
all_feats = eng.get_feature_columns(df_full, tier=2)
print(f"All features: {len(all_feats)}")

# ═══════════════════════════════════════════════════
#  2. FEATURE SET DEFINITIONS (핵심!)
# ═══════════════════════════════════════════════════
# Missing value imputation: group-median for all models (Strategy A from V7)
MS = {'멕시코','사우디아라비야','아랍에미리트','카자흐스탄','튀르키예','ETC'}
MH = {'몽골','사우디아라비야','싱가포르','필리핀','홍콩','ETC'}
SF = ['revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
      'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag','spend_x_fx']
HF = ['hallyu_index','hallyu_x_interest']
IF_cols = ['image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
      'image_kdrama_pct','culture_affinity','culture_x_interest']

def impute_groupmedian(d):
    """Group-median imputation (same as Strategy A)"""
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

# ─── Model A: Lag / History ───
# "과거 방문 패턴이 미래를 예측한다"
FEATS_A = [
    # Visitor lags & moving averages
    'visitor_lag_1', 'visitor_lag_2', 'visitor_lag_3', 'visitor_lag_6', 'visitor_lag_12',
    'visitor_ma_3', 'visitor_ma_6', 'visitor_ma_12',
    'visitor_yoy_growth',
    'recovery_ratio',
    # Group dummies (structural)
    'group_A1_western_surge', 'group_A2_asia_surge', 'group_B_recovering',
    'group_C_underperform', 'group_ETC_group',
    # Minimal time signal (just month for seasonality anchor)
    'month_sin', 'month_cos',
]
FEATS_A = [f for f in FEATS_A if f in df_imp.columns]

# ─── Model B: Macro / Economic ───
# "경제 환경이 방문 수를 결정한다"
FEATS_B = [
    # Exchange rate family
    'exchange_rate', 'exchange_rate_lag_1', 'exchange_rate_ma_3',
    'exchange_rate_change_pct', 'exchange_rate_volatility',
    # Oil price family
    'oil_price', 'oil_price_lag_1', 'oil_price_ma_3',
    # Travel interest (demand signal)
    'travel_interest_pct', 'travel_interest_lag_1', 'travel_interest_lag_2', 'travel_interest_lag_3',
    'global_travel_interest_pct', 'global_interest_lag_1', 'global_interest_lag_2', 'global_interest_lag_3',
    # Spending × FX interaction
    'spend_x_fx',
    # COVID regime shift
    'covid_period', 'post_covid',
    # Group dummies (structural)
    'group_A1_western_surge', 'group_A2_asia_surge', 'group_B_recovering',
    'group_C_underperform', 'group_ETC_group',
    # Single lag anchor (visitor_lag_12 = same month last year, minimal overlap)
    'visitor_lag_12',
]
FEATS_B = [f for f in FEATS_B if f in df_imp.columns]

# ─── Model C: Event / Season / Culture ───
# "계절성, 이벤트, 문화적 매력이 방문을 끌어당긴다"
FEATS_C = [
    # Seasonality (full)
    'month_sin', 'month_cos', 'quarter', 'is_peak_season', 'is_summer', 'is_winter',
    # Weather
    'weather_comfort', 'temp_lag_1', 'precip_lag_1', 'temp_anomaly', 'heavy_rain',
    # Events
    'event_national_lag_1', 'event_seoul_lag_1', 'event_seoul_ratio',
    'event_intensity', 'event_weighted',
    # Culture / Image
    'hallyu_index', 'hallyu_x_interest',
    'image_top1_pct', 'image_kfood_pct', 'image_kpop_pct', 'image_kbeauty_pct',
    'image_kdrama_pct', 'culture_affinity', 'culture_x_interest',
    # Satisfaction (demand quality signal)
    'revisit_rate_lag', 'stay_days_lag', 'per_capita_spend_usd_lag', 'daily_spend_usd_lag',
    'overall_satisfaction_lag', 'revisit_intention_lag', 'recommend_intention_lag',
    # Purpose
    'tourism_dominant', 'business_ratio',
    # COVID regime
    'covid_period', 'post_covid',
    # Group dummies
    'group_A1_western_surge', 'group_A2_asia_surge', 'group_B_recovering',
    'group_C_underperform', 'group_ETC_group',
    # Single lag anchor (visitor_lag_12)
    'visitor_lag_12',
]
FEATS_C = [f for f in FEATS_C if f in df_imp.columns]

print(f"\n{'='*60}")
print(f"  V8 FEATURE SETS (Information Diversity)")
print(f"{'='*60}")
print(f"  Model A (Lag/History):       {len(FEATS_A)} features")
print(f"  Model B (Macro/Economic):    {len(FEATS_B)} features")
print(f"  Model C (Event/Season/Cult): {len(FEATS_C)} features")

# Check overlap
set_a, set_b, set_c = set(FEATS_A), set(FEATS_B), set(FEATS_C)
ab = set_a & set_b; ac = set_a & set_c; bc = set_b & set_c
a_only = set_a - set_b - set_c
b_only = set_b - set_a - set_c
c_only = set_c - set_a - set_b
print(f"\n  Feature Overlap:")
print(f"    A-only:  {len(a_only)} ({', '.join(sorted(a_only)[:5])}...)")
print(f"    B-only:  {len(b_only)} ({', '.join(sorted(b_only)[:5])}...)")
print(f"    C-only:  {len(c_only)} ({', '.join(sorted(c_only)[:5])}...)")
print(f"    A∩B:     {len(ab)} ({', '.join(sorted(ab))})")
print(f"    A∩C:     {len(ac)} ({', '.join(sorted(ac))})")
print(f"    B∩C:     {len(bc)} ({', '.join(sorted(bc))})")
shared_all = set_a & set_b & set_c
print(f"    A∩B∩C:   {len(shared_all)} ({', '.join(sorted(shared_all))})")

# ═══════════════════════════════════════════════════
#  3. TRAIN & EVALUATE (per model)
# ═══════════════════════════════════════════════════
def train_model(df, feats, label, train_end='2023-12', test_start='2024-01', test_end='2024-12',
                predict_start='2025-01', predict_end='2025-12'):
    """Train GBM+Ridge+WLag → StackedEnsemble per group, return test & pred results"""
    train = df[(df['year_month'] >= TRAIN_START) & (df['year_month'] <= train_end)].copy()
    covid = (train['year_month'] >= COVID_S) & (train['year_month'] <= COVID_E)
    train = train[~covid]
    test = df[(df['year_month'] >= test_start) & (df['year_month'] <= test_end)].copy()
    future = df[(df['year_month'] >= predict_start) & (df['year_month'] <= predict_end)].copy()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Features: {len(feats)}")
    print(f"  Train: {len(train)} | Test: {len(test)} | Future: {len(future)}")
    print(f"{'='*60}")

    all_test, all_pred = [], []

    for group in sorted(train['country_group'].unique()):
        gt = train[train['country_group'] == group]
        gv = test[test['country_group'] == group]
        gf = future[future['country_group'] == group]

        vf = [c for c in feats if c in gt.columns]

        Xt = gt[vf].values.astype(float); yt = gt['visitors'].values.astype(float)
        Xv = gv[vf].values.astype(float); yv = gv['visitors'].values.astype(float)
        meta_v = gv[['year_month','country']].copy()

        ok = ~np.isnan(yt) & (yt > 0); Xt, yt = Xt[ok], yt[ok]
        ok2 = ~np.isnan(yv) & (yv > 0); Xv, yv = Xv[ok2], yv[ok2]
        meta_v = meta_v.iloc[ok2.nonzero()[0]]

        # Fill NaN in features with 0
        Xt = np.nan_to_num(Xt, nan=0.0)
        Xv = np.nan_to_num(Xv, nan=0.0)

        if len(Xt) < 10 or len(Xv) == 0:
            print(f"  {group}: SKIP (train={len(Xt)})")
            continue

        # 3 Models
        models = {
            'GBM': GradientBoostingV4(n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=5),
            'Ridge': RidgeRegression(alpha=10.0),
            'WLag': WeightedLagModel(),
        }
        fitted = {}
        model_results = {}
        for n, m in models.items():
            try:
                m.fit(Xt, yt, feature_names=vf); fitted[n] = m
                p = np.maximum(m.predict(Xv), 0)
                model_results[n] = (rmse(yv, p), mape(yv, p), p)
            except Exception as ex:
                print(f"    {group}/{n}: FAIL ({ex})")

        # Stacked Ensemble
        if len(fitted) >= 2:
            st = StackedEnsemble(list(fitted.values()), list(fitted.keys()))
            st.fit_weights(Xv, yv)
            sp = np.maximum(st.predict(Xv), 0)
            model_results['Stacked'] = (rmse(yv, sp), mape(yv, sp), sp)
            fitted['Stacked'] = st

        if not model_results: continue
        best_n = min(model_results, key=lambda k: model_results[k][0])
        best_r, best_m, best_p = model_results[best_n]
        print(f"  {group}: best={best_n}, RMSE={best_r:,.0f}, MAPE={best_m:.1f}%")

        # Test
        tdf = meta_v.copy(); tdf['actual'] = yv; tdf['predicted'] = best_p
        tdf['country_group'] = group; tdf['model'] = best_n
        all_test.append(tdf)

        # Future
        if len(gf) > 0:
            Xf = gf[vf].values.astype(float)
            yf_actual = gf['visitors'].values.astype(float)
            meta_f = gf[['year_month','country']].copy()
            Xf = np.nan_to_num(Xf, nan=0.0)
            fp = np.maximum(fitted[best_n].predict(Xf), 0)
            pdf = meta_f.copy()
            pdf['actual'] = yf_actual
            pdf['predicted'] = fp
            pdf['country_group'] = group; pdf['model'] = best_n
            all_pred.append(pdf)

    test_results = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    pred_results = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    return test_results, pred_results


# ═══════════════════════════════════════════════════
#  4. RUN ALL 3 MODELS
# ═══════════════════════════════════════════════════
print("\n" + "#"*60)
print("# MODEL A: Lag / History (과거 방문 패턴)")
print("#"*60)
testA, predA = train_model(df_imp, FEATS_A, "Model A: Lag/History")

print("\n" + "#"*60)
print("# MODEL B: Macro / Economic (경제 환경)")
print("#"*60)
testB, predB = train_model(df_imp, FEATS_B, "Model B: Macro/Economic")

print("\n" + "#"*60)
print("# MODEL C: Event / Season / Culture (수요 동인)")
print("#"*60)
testC, predC = train_model(df_imp, FEATS_C, "Model C: Event/Season/Culture")


# ═══════════════════════════════════════════════════
#  5. PREDICTION CORRELATION CHECK (교수님 기준)
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  PREDICTION CORRELATION CHECK")
print("  (교수님 기준: <0.5 good, 0.6-0.8 borderline, >0.9 stacking 무의미)")
print("="*70)

def merge_predictions(rA, rB, rC, col='predicted'):
    """Merge predictions from 3 models on (year_month, country)"""
    base = rA[['year_month','country',col]].rename(columns={col:'pred_A'})
    bm = rB[['year_month','country',col]].rename(columns={col:'pred_B'})
    cm = rC[['year_month','country',col]].rename(columns={col:'pred_C'})
    merged = base.merge(bm, on=['year_month','country'], how='inner')
    merged = merged.merge(cm, on=['year_month','country'], how='inner')
    return merged

# Test predictions correlation
if not testA.empty and not testB.empty and not testC.empty:
    mp = merge_predictions(testA, testB, testC)
    if len(mp) > 10:
        corr_AB = np.corrcoef(mp['pred_A'], mp['pred_B'])[0,1]
        corr_AC = np.corrcoef(mp['pred_A'], mp['pred_C'])[0,1]
        corr_BC = np.corrcoef(mp['pred_B'], mp['pred_C'])[0,1]

        print(f"\n  Test Prediction Correlations:")
        print(f"    A↔B (Lag vs Macro):    {corr_AB:.4f}")
        print(f"    A↔C (Lag vs Event):    {corr_AC:.4f}")
        print(f"    B↔C (Macro vs Event):  {corr_BC:.4f}")

        # Error correlation
        me = mp.copy()
        me = me.merge(testA[['year_month','country','actual']], on=['year_month','country'], how='left')
        me['err_A'] = me['pred_A'] - me['actual']
        me['err_B'] = me['pred_B'] - me['actual']
        me['err_C'] = me['pred_C'] - me['actual']

        ecorr_AB = np.corrcoef(me['err_A'], me['err_B'])[0,1]
        ecorr_AC = np.corrcoef(me['err_A'], me['err_C'])[0,1]
        ecorr_BC = np.corrcoef(me['err_B'], me['err_C'])[0,1]

        print(f"\n  Error Correlations:")
        print(f"    A↔B: {ecorr_AB:.4f}")
        print(f"    A↔C: {ecorr_AC:.4f}")
        print(f"    B↔C: {ecorr_BC:.4f}")

        avg_pred_corr = (abs(corr_AB) + abs(corr_AC) + abs(corr_BC)) / 3
        avg_err_corr = (abs(ecorr_AB) + abs(ecorr_AC) + abs(ecorr_BC)) / 3

        print(f"\n  Average |Prediction Corr|: {avg_pred_corr:.4f}")
        print(f"  Average |Error Corr|:      {avg_err_corr:.4f}")

        if avg_pred_corr > 0.9 and avg_err_corr > 0.9:
            print("  → ❌ Pred & Error 모두 >0.9: Stacking 의미 없음 (V5~V7과 동일)")
            DO_STACK = False
        elif avg_pred_corr > 0.9 and avg_err_corr <= 0.9:
            print("  → ⚠️ Pred >0.9 but Error corr 낮음 — 에러 패턴 다양성 있음! Stacking 시도")
            DO_STACK = True
        elif avg_pred_corr > 0.6:
            print("  → ⚠️ 0.6~0.9: Borderline — stacking 시도해볼 가치 있음")
            DO_STACK = True
        else:
            print("  → ✅ <0.6: Good diversity — stacking 효과 기대!")
            DO_STACK = True
    else:
        print("  Not enough overlapping predictions for correlation check.")
        DO_STACK = False
else:
    print("  One or more models produced no test results.")
    DO_STACK = False


# ═══════════════════════════════════════════════════
#  6. META-STACKING (2nd Level)
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  META-STACKING (2nd Level)")
print("="*70)

if DO_STACK and not testA.empty and not testB.empty and not testC.empty:
    # Merge all test predictions with actual
    meta_test = testA[['year_month','country','country_group','actual']].copy()
    meta_test = meta_test.merge(testA[['year_month','country','predicted']].rename(columns={'predicted':'pred_A'}),
                                 on=['year_month','country'])
    meta_test = meta_test.merge(testB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),
                                 on=['year_month','country'], how='inner')
    meta_test = meta_test.merge(testC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),
                                 on=['year_month','country'], how='inner')

    print(f"  Meta-train samples: {len(meta_test)}")

    # Simple meta-learners (Ridge implemented manually — no sklearn needed)
    X_meta = meta_test[['pred_A','pred_B','pred_C']].values
    y_meta = meta_test['actual'].values

    # Ridge meta (closed-form: w = (X'X + αI)^{-1} X'y with intercept via centering)
    class SimpleRidge:
        def __init__(self, alpha=1.0):
            self.alpha = alpha
        def fit(self, X, y):
            self.x_mean = X.mean(axis=0)
            self.y_mean = y.mean()
            Xc = X - self.x_mean
            yc = y - self.y_mean
            A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
            self.intercept_ = self.y_mean - self.x_mean @ self.coef_
        def predict(self, X):
            return X @ self.coef_ + self.intercept_

    ridge_meta = SimpleRidge(alpha=1.0)
    ridge_meta.fit(X_meta, y_meta)
    meta_pred_ridge = np.maximum(ridge_meta.predict(X_meta), 0)

    # Simple average
    meta_pred_avg = (meta_test['pred_A'] + meta_test['pred_B'] + meta_test['pred_C']) / 3
    meta_pred_avg = np.maximum(meta_pred_avg.values, 0)

    # Weighted average (inverse MAPE weighting)
    mape_a = np.mean(np.abs((meta_test['actual'] - meta_test['pred_A']) / meta_test['actual']))
    mape_b = np.mean(np.abs((meta_test['actual'] - meta_test['pred_B']) / meta_test['actual']))
    mape_c = np.mean(np.abs((meta_test['actual'] - meta_test['pred_C']) / meta_test['actual']))
    inv_mapes = np.array([1/mape_a, 1/mape_b, 1/mape_c])
    weights = inv_mapes / inv_mapes.sum()
    meta_pred_wmean = np.maximum(
        meta_test['pred_A'].values * weights[0] +
        meta_test['pred_B'].values * weights[1] +
        meta_test['pred_C'].values * weights[2], 0)

    print(f"\n  Meta-Learner Weights:")
    print(f"    Ridge coefficients: A={ridge_meta.coef_[0]:.4f}, B={ridge_meta.coef_[1]:.4f}, C={ridge_meta.coef_[2]:.4f}")
    print(f"    Ridge intercept: {ridge_meta.intercept_:.2f}")
    print(f"    Inverse-MAPE weights: A={weights[0]:.4f}, B={weights[1]:.4f}, C={weights[2]:.4f}")

    # Evaluate all approaches on test
    def eval_meta(name, pred, actual):
        r = np.sqrt(np.mean((actual - pred)**2))
        m = np.mean(np.abs((actual - pred) / actual)) * 100
        te = abs(actual.sum() - pred.sum()) / actual.sum() * 100
        print(f"    {name:<25s}  RMSE={r:>10,.0f}  MAPE={m:>6.1f}%  TotalErr={te:>5.1f}%")
        return {'name': name, 'rmse': r, 'mape': m, 'total_err': te, 'pred': pred}

    print(f"\n  Test Performance Comparison:")
    results = []
    results.append(eval_meta("Model A (Lag)", meta_test['pred_A'].values, y_meta))
    results.append(eval_meta("Model B (Macro)", meta_test['pred_B'].values, y_meta))
    results.append(eval_meta("Model C (Event)", meta_test['pred_C'].values, y_meta))
    results.append(eval_meta("Simple Average", meta_pred_avg, y_meta))
    results.append(eval_meta("Weighted Average", meta_pred_wmean, y_meta))
    results.append(eval_meta("Ridge Meta", meta_pred_ridge, y_meta))

    # Best approach
    best = min(results, key=lambda r: r['rmse'])
    print(f"\n  🏆 Best: {best['name']} (RMSE={best['rmse']:,.0f}, MAPE={best['mape']:.1f}%)")

    # ─── Apply best meta to 2025 predictions ───
    if not predA.empty and not predB.empty and not predC.empty:
        meta_future = predA[['year_month','country','country_group','actual']].copy()
        meta_future = meta_future.merge(
            predA[['year_month','country','predicted']].rename(columns={'predicted':'pred_A'}),
            on=['year_month','country'])
        meta_future = meta_future.merge(
            predB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),
            on=['year_month','country'], how='inner')
        meta_future = meta_future.merge(
            predC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),
            on=['year_month','country'], how='inner')

        Xf_meta = meta_future[['pred_A','pred_B','pred_C']].values

        meta_future['meta_ridge'] = np.maximum(ridge_meta.predict(Xf_meta), 0)
        meta_future['meta_avg'] = np.maximum((meta_future['pred_A'] + meta_future['pred_B'] + meta_future['pred_C']) / 3, 0)
        meta_future['meta_wmean'] = np.maximum(
            meta_future['pred_A'] * weights[0] +
            meta_future['pred_B'] * weights[1] +
            meta_future['pred_C'] * weights[2], 0)

        print(f"\n  2025 Predictions (Meta-Stacked):")
        print(f"  {'Method':<20s} {'Total':>15s}")
        for col, name in [('pred_A','Model A'), ('pred_B','Model B'), ('pred_C','Model C'),
                          ('meta_avg','Simple Avg'), ('meta_wmean','Weighted Avg'), ('meta_ridge','Ridge Meta')]:
            total = meta_future[col].sum()
            print(f"  {name:<20s} {total:>15,.0f}")

        # Monthly breakdown for best
        best_col = {'Simple Average':'meta_avg', 'Weighted Average':'meta_wmean', 'Ridge Meta':'meta_ridge',
                    'Model A (Lag)':'pred_A', 'Model B (Macro)':'pred_B', 'Model C (Event)':'pred_C'}
        bc = best_col.get(best['name'], 'meta_ridge')

        print(f"\n  Monthly Breakdown (Best: {best['name']}):")
        monthly = meta_future.groupby('year_month').agg({
            'pred_A':'sum', 'pred_B':'sum', 'pred_C':'sum',
            'meta_ridge':'sum', 'meta_avg':'sum', 'meta_wmean':'sum',
            'actual':'sum'
        }).reset_index()

        for _, row in monthly.iterrows():
            a_str = "{:,.0f}".format(row['actual']) if row['actual'] > 0 else "N/A"
            print(f"    {row['year_month']}: A={row['pred_A']:>10,.0f}  B={row['pred_B']:>10,.0f}  C={row['pred_C']:>10,.0f}  Meta={row[bc]:>10,.0f}  Actual={a_str}")

        # Save
        meta_future.to_csv(os.path.join(PROJECT_ROOT, 'outputs', 'v8_meta_predictions.csv'), index=False, encoding='utf-8-sig')

else:
    if not DO_STACK:
        print("  Skipping meta-stacking: correlation too high (same problem as V5~V7)")
    else:
        print("  Skipping: insufficient data from individual models")


# ═══════════════════════════════════════════════════
#  7. INDIVIDUAL MODEL SUMMARIES
# ═══════════════════════════════════════════════════
print("\n" + "="*70)
print("  V8 INDIVIDUAL MODEL TEST RESULTS (2024)")
print("="*70)

def summarize(results, label):
    if results.empty: return {}
    r24 = results[results['country_group'] != 'ETC_group']
    rmse_24 = np.sqrt(np.mean((r24['actual'] - r24['predicted'])**2))
    mape_24 = np.mean(np.abs((r24['actual'] - r24['predicted']) / r24['actual'])) * 100
    m24 = r24.groupby('year_month').agg({'actual':'sum','predicted':'sum'})
    terr = abs(m24['actual'].sum() - m24['predicted'].sum()) / m24['actual'].sum() * 100
    mall = results.groupby('year_month').agg({'actual':'sum','predicted':'sum'})
    gerr = abs(mall['actual'].sum() - mall['predicted'].sum()) / mall['actual'].sum() * 100

    print(f"\n  {label}")
    print(f"    24-Country MAPE: {mape_24:.1f}%, RMSE: {rmse_24:,.0f}")
    print(f"    24-Country Total Err: {terr:.1f}%")
    print(f"    Grand Total Err: {gerr:.1f}%")
    return {'label':label, 'mape':mape_24, 'rmse':rmse_24, 'terr':terr, 'gerr':gerr}

sA = summarize(testA, "Model A: Lag/History")
sB = summarize(testB, "Model B: Macro/Economic")
sC = summarize(testC, "Model C: Event/Season/Culture")

# Save individual results
for label, test, pred in [('A', testA, predA), ('B', testB, predB), ('C', testC, predC)]:
    if not test.empty:
        test.to_csv(os.path.join(PROJECT_ROOT, 'outputs', f'v8_test_{label}.csv'), index=False, encoding='utf-8-sig')
    if not pred.empty:
        pred.to_csv(os.path.join(PROJECT_ROOT, 'outputs', f'v8_pred_{label}.csv'), index=False, encoding='utf-8-sig')

print("\n✅ V8 Complete!")
