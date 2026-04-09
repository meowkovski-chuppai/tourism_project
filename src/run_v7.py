"""
V7: Updated Parameters
- Train: 2018-01 ~ 2023-12
- COVID exclusion: 2020-01 ~ 2023-06
- Test: 2024-01 ~ 2024-12
- Predict: 2025-01 ~ 2025-12
- 3 Missing-value strategies (A/B/C) + ETC country
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

# ═══ Load ═══
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'country')
bp = DATA_DIR
print(f"Base: {bp}")

loader = V4DataLoader(bp); panel = loader.build_panel()

# ETC
gdir = loader.global_dir; gc=None
for e in os.scandir(gdir):
    if '외래관광객 추이' in _nfc(e.name) and e.name.endswith('.csv'): gc=e.path; break
gdf = _read_csv_safe(gc); gdf.columns=[_nfc(c) for c in gdf.columns]
gdf['year_month']=gdf['기준년월'].astype(str).apply(_normalize_yearmonth)
gdf['total_visitors']=pd.to_numeric(gdf['방한 외래관광객'].astype(str).str.replace(',',''),errors='coerce')
gdf['exchange_rate']=pd.to_numeric(gdf['환율(원)'].astype(str).str.replace(',',''),errors='coerce')
gdf['oil_price']=pd.to_numeric(gdf['국제유가(달러)'].astype(str).str.replace(',',''),errors='coerce')
s24=panel.groupby('year_month')['visitors'].sum().reset_index().rename(columns={'visitors':'s24'})
edf=gdf[['year_month','total_visitors','exchange_rate','oil_price']].merge(s24,on='year_month',how='left')
edf['visitors']=edf['total_visitors']-edf['s24'].fillna(0)
edf['country']='ETC';edf['country_group']='ETC_group'
edf['year']=edf['year_month'].str[:4].astype(int);edf['month']=edf['year_month'].str[5:7].astype(int)
edf=edf[['year_month','country','country_group','visitors','exchange_rate','oil_price','year','month']]
pe=pd.concat([panel,edf],ignore_index=True).sort_values(['country','year_month']).reset_index(drop=True)
print(f"Panel: {pe.shape[0]} rows, {pe['country'].nunique()} countries")

eng=V4FeatureEngineer(include_tier2=True)
df_full=eng.build_features(pe)
base_feats=eng.get_feature_columns(df_full,tier=2)
print(f"Features: {len(base_feats)}")

# ═══ Missing value definitions ═══
MS={'멕시코','사우디아라비야','아랍에미리트','카자흐스탄','튀르키예','ETC'}
MH={'몽골','사우디아라비야','싱가포르','필리핀','홍콩','ETC'}
SF=['revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
    'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag','spend_x_fx']
HF=['hallyu_index','hallyu_x_interest']
IF=['image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
    'image_kdrama_pct','culture_affinity','culture_x_interest']

def mk_A(d,f):
    d=d.copy()
    for col in SF+HF+IF:
        if col not in d.columns: continue
        ms=MS if col in SF else MH
        for c in ms:
            m=d['country']==c
            if not m.any(): continue
            g=d.loc[m,'country_group'].iloc[0]
            gm=(d['country_group']==g)&(~d['country'].isin(ms))
            d.loc[m,col]=d.loc[m,col].fillna(d.loc[gm,col].median())
    return d,f

def mk_B(d,f):
    d=d.copy();ge={}
    for g,cs in COUNTRY_GROUPS.items():
        ex=set()
        for c in cs:
            if c in MS: ex.update(SF)
            if c in MH: ex.update(HF);ex.update(IF)
        ge[g]=ex
    ge['ETC_group']=set(SF+HF+IF)
    return d,f,ge

def mk_C(d,f):
    d=d.copy()
    d['has_sat']=(~d['country'].isin(MS)).astype(int)
    d['has_hal']=(~d['country'].isin(MH)).astype(int)
    d['has_img']=(~d['country'].isin(MH)).astype(int)
    for col in SF+HF+IF:
        if col in d.columns: d[col]=d[col].fillna(d[col].median())
    return d,f+['has_sat','has_hal','has_img']

# ═══ Train & Evaluate function ═══
def train_eval(df, feats, label, train_end, test_start, test_end, 
               predict_start=None, predict_end=None, group_excl=None):
    """Train, evaluate on test, optionally predict future"""
    train = df[(df['year_month']>=TRAIN_START)&(df['year_month']<=train_end)].copy()
    covid = (train['year_month']>=COVID_S)&(train['year_month']<=COVID_E)
    train = train[~covid]
    test = df[(df['year_month']>=test_start)&(df['year_month']<=test_end)].copy()
    
    if predict_start and predict_end:
        future = df[(df['year_month']>=predict_start)&(df['year_month']<=predict_end)].copy()
    else:
        future = None
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train: {len(train)} rows (2018~{train_end}, excl COVID)")
    print(f"  Test: {len(test)} rows ({test_start}~{test_end})")
    if future is not None:
        print(f"  Predict: {len(future)} rows ({predict_start}~{predict_end})")
    print(f"{'='*60}")
    
    all_test = []
    all_pred = []
    
    for group in sorted(train['country_group'].unique()):
        gt = train[train['country_group']==group]
        gv = test[test['country_group']==group]
        
        vf = [c for c in feats if c in gt.columns]
        if group_excl and group in group_excl:
            vf = [c for c in vf if c not in group_excl[group]]
        
        Xt=gt[vf].values.astype(float); yt=gt['visitors'].values.astype(float)
        Xv=gv[vf].values.astype(float); yv=gv['visitors'].values.astype(float)
        meta_v=gv[['year_month','country']].copy()
        
        ok=~np.isnan(yt)&(yt>0); Xt,yt=Xt[ok],yt[ok]
        ok2=~np.isnan(yv)&(yv>0); Xv,yv=Xv[ok2],yv[ok2]
        meta_v=meta_v.iloc[ok2.nonzero()[0]]
        
        if len(Xt)<10 or len(Xv)==0: 
            print(f"  {group}: SKIP (train={len(Xt)})")
            continue
        
        # Models
        models = {
            'GBM': GradientBoostingV4(n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=5),
            'Ridge': RidgeRegression(alpha=10.0),
            'WLag': WeightedLagModel(),
        }
        fitted = {}
        model_results = {}
        for n,m in models.items():
            try:
                m.fit(Xt,yt,feature_names=vf); fitted[n]=m
                p=np.maximum(m.predict(Xv),0)
                model_results[n]=(rmse(yv,p), mape(yv,p), p)
            except: pass
        
        # Stacked
        if len(fitted)>=2:
            st=StackedEnsemble(list(fitted.values()),list(fitted.keys()))
            st.fit_weights(Xv,yv)
            sp=np.maximum(st.predict(Xv),0)
            model_results['Stacked']=(rmse(yv,sp),mape(yv,sp),sp)
            fitted['Stacked']=st
        
        if not model_results: continue
        best_n=min(model_results, key=lambda k: model_results[k][0])
        best_r,best_m,best_p=model_results[best_n]
        print(f"  {group}: best={best_n}, RMSE={best_r:,.0f}, MAPE={best_m:.1f}%, feats={len(vf)}")
        
        # Test results
        tdf=meta_v.copy(); tdf['actual']=yv; tdf['predicted']=best_p
        tdf['country_group']=group; tdf['model']=best_n
        all_test.append(tdf)
        
        # Future predictions
        if future is not None:
            gf = future[future['country_group']==group]
            if len(gf)>0:
                Xf=gf[vf].values.astype(float)
                yf_actual=gf['visitors'].values.astype(float)
                meta_f=gf[['year_month','country']].copy()
                # Fill NaN in future features with 0 (missing survey data etc)
                Xf = np.nan_to_num(Xf, nan=0.0)
                ok3=np.ones(len(Xf), dtype=bool)  # keep all rows
                if ok3.sum()>0:
                    Xf_clean=Xf[ok3]
                    best_model=fitted[best_n]
                    fp=np.maximum(best_model.predict(Xf_clean),0)
                    pdf=meta_f.iloc[ok3.nonzero()[0]].copy()
                    pdf['actual']=yf_actual[ok3]
                    pdf['predicted']=fp
                    pdf['country_group']=group; pdf['model']=best_n
                    all_pred.append(pdf)
    
    test_results = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    pred_results = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    return test_results, pred_results

# ═══ Summarize function ═══
def summarize(results, label):
    if results.empty: return {}
    r24=results[results['country_group']!='ETC_group']
    
    rmse_24=np.sqrt(np.mean((r24['actual']-r24['predicted'])**2))
    mape_24=np.mean(np.abs((r24['actual']-r24['predicted'])/r24['actual']))*100
    
    m24=r24.groupby('year_month').agg({'actual':'sum','predicted':'sum'})
    terr=abs(m24['actual'].sum()-m24['predicted'].sum())/m24['actual'].sum()*100
    
    mall=results.groupby('year_month').agg({'actual':'sum','predicted':'sum'})
    gerr=abs(mall['actual'].sum()-mall['predicted'].sum())/mall['actual'].sum()*100
    
    # ETC
    retc=results[results['country_group']=='ETC_group']
    etc_mape=np.mean(np.abs((retc['actual']-retc['predicted'])/retc['actual']))*100 if len(retc)>0 else None
    
    # Per group
    gstats={}
    for g in sorted(results['country_group'].unique()):
        gd=results[results['country_group']==g]
        gr=np.sqrt(np.mean((gd['actual']-gd['predicted'])**2))
        gm=np.mean(np.abs((gd['actual']-gd['predicted'])/gd['actual']))*100
        gstats[g]=(gr,gm)
    
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  {'Group':<22s} {'RMSE':>10s} {'MAPE':>8s}")
    for g in sorted(gstats):
        r,m=gstats[g]
        print(f"  {g:<22s} {r:>10,.0f} {m:>7.1f}%")
    print(f"  {'─'*42}")
    print(f"  {'24-Country':<22s} {rmse_24:>10,.0f} {mape_24:>7.1f}%")
    print(f"  {'24C Total Err':<22s} {'':>10s} {terr:>7.1f}%")
    if etc_mape: print(f"  {'ETC':<22s} {'':>10s} {etc_mape:>7.1f}%")
    print(f"  {'Grand Total Err':<22s} {'':>10s} {gerr:>7.1f}%")
    
    return {'label':label,'mape':mape_24,'rmse':rmse_24,'terr':terr,'gerr':gerr,'etc':etc_mape,'gstats':gstats}

# ═══ Run 3 models ═══
print("\n" + "#"*60)
print("# MODEL A: Group-Median Imputation")
print("#"*60)
dA,fA=mk_A(df_full.copy(),list(base_feats))
testA,predA=train_eval(dA,fA,"Model A",
    train_end='2023-12', test_start='2024-01', test_end='2024-12',
    predict_start='2025-01', predict_end='2025-12')

print("\n" + "#"*60)
print("# MODEL B: Drop Missing Features")
print("#"*60)
dB,fB,eB=mk_B(df_full.copy(),list(base_feats))
testB,predB=train_eval(dB,fB,"Model B",
    train_end='2023-12', test_start='2024-01', test_end='2024-12',
    predict_start='2025-01', predict_end='2025-12', group_excl=eB)

print("\n" + "#"*60)
print("# MODEL C: Missingness Indicators")
print("#"*60)
dC,fC=mk_C(df_full.copy(),list(base_feats))
testC,predC=train_eval(dC,fC,"Model C",
    train_end='2023-12', test_start='2024-01', test_end='2024-12',
    predict_start='2025-01', predict_end='2025-12')

# ═══ Compare Test (2024) ═══
print("\n" + "="*70)
print("  V7 TEST RESULTS: 2024 (Train 2018~2023, excl COVID 2020-01~2023-06)")
print("="*70)

sA=summarize(testA,"Model A: Imputation")
sB=summarize(testB,"Model B: Drop Features")
sC=summarize(testC,"Model C: Indicators")

print("\n" + "="*70)
print("  RANKING")
print("="*70)
stats=[s for s in [sA,sB,sC] if s]
by_mape=sorted(stats,key=lambda s:s['mape'])
mape_str=' > '.join('{} ({:.2f}%)'.format(s['label'],s['mape']) for s in by_mape)
print(f'\n  By MAPE: {mape_str}')
by_gerr=sorted(stats,key=lambda s:s['gerr'])
gerr_str=' > '.join('{} ({:.1f}%)'.format(s['label'],s['gerr']) for s in by_gerr)
print('  By Grand Err: ' + gerr_str)
by_rmse=sorted(stats,key=lambda s:s['rmse'])
rmse_str=' > '.join('{} ({:,.0f})'.format(s['label'],s['rmse']) for s in by_rmse)
print('  By RMSE: ' + rmse_str)

# ═══ 2025 Predictions ═══
print("\n" + "="*70)
print("  2025 PREDICTIONS")
print("="*70)

for label,pred in [('A',predA),('B',predB),('C',predC)]:
    if pred.empty: 
        print(f"  Model {label}: no predictions")
        continue
    has_actual = pred['actual'].notna() & (pred['actual']>0)
    if has_actual.sum()>0:
        p_valid=pred[has_actual]
        mp=np.mean(np.abs((p_valid['actual']-p_valid['predicted'])/p_valid['actual']))*100
        total_a=p_valid.groupby('year_month')['actual'].sum().sum()
        total_p=p_valid.groupby('year_month')['predicted'].sum().sum()
        te=abs(total_a-total_p)/total_a*100
        print(f"  Model {label}: {len(p_valid)} rows with actual data, MAPE={mp:.1f}%, Total Err={te:.1f}%")
        print(f"    Actual total: {total_a:,.0f}, Predicted: {total_p:,.0f}")
    else:
        total_p=pred['predicted'].sum()
        print(f"  Model {label}: {len(pred)} rows, Predicted total: {total_p:,.0f} (no actual for comparison)")
    
    # Monthly breakdown
    monthly=pred.groupby('year_month').agg({'predicted':'sum','actual':'sum'}).reset_index()
    print(f"    Monthly predictions:")
    for _,row in monthly.iterrows():
        a_str=f"{row['actual']:,.0f}" if row['actual']>0 else "N/A"
        print(f"      {row['year_month']}: pred={row['predicted']:,.0f}, actual={a_str}")

# Save
for label,test,pred in [('A',testA,predA),('B',testB,predB),('C',testC,predC)]:
    if not test.empty:
        test.to_csv(os.path.join(PROJECT_ROOT, 'outputs', f'v7_test_{label}.csv'),index=False,encoding='utf-8-sig')
    if not pred.empty:
        pred.to_csv(os.path.join(PROJECT_ROOT, 'outputs', f'v7_pred_{label}.csv'),index=False,encoding='utf-8-sig')

print("\nAll saved!")
