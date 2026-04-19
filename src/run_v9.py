"""
V9 Final: Bootstrap×10 + BayesianRidge(log) Meta-Learner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실험 결과 요약:
  - Bootstrap: sub-model variance 감소 (RMSE 개선)
  - BayesianRidge(log) meta: 소규모 국가 예측 개선 (MAPE 29.6%→14.9%)
  - 조합: BS+BRlog = MAPE 14.9% (best)
  - Country FE, Mixed Effects 실험 결과 기존 V9 구조(per-group)가 최적

구조:
  Level 1 (내부 스태킹): GBM+Ridge+WLag → 역RMSE 가중평균 (per country_group)
  Bootstrap: 10회 복원추출 → 10개 Stacked 모델 → 예측 평균
  Level 2 (메타러너): BayesianRidge on log1p(predictions) → 최종 결합

출력:
  1. OOF 성능 (4-fold rolling window)
  2. 2024 Test 성능
  3. 2025 예측 (국가별 × 월별)
"""
import sys, os, unicodedata, time, warnings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge as SkBayesianRidge
from loader import V4DataLoader, _read_csv_safe, _nfc, _normalize_yearmonth
from engineer import V4FeatureEngineer
warnings.filterwarnings('ignore')
np.random.seed(42)

N_TREES=80; LR=0.05; MAX_DEPTH=2; MIN_LEAF=5; SUBSAMPLE=0.8
RIDGE_ALPHA=10.0; N_BOOTSTRAP=int(sys.argv[1]) if len(sys.argv)>1 else 10

t0=time.time()
print(f"V9 Final: Bootstrap×{N_BOOTSTRAP} + BayesianRidge(log)")
print(f"Config: trees={N_TREES}, depth={MAX_DEPTH}, leaf={MIN_LEAF}, bs={N_BOOTSTRAP}")

# ══════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════
def _imp(X):
    X=X.copy()
    for j in range(X.shape[1]):
        m=np.isnan(X[:,j])
        if m.any(): X[m,j]=np.nanmedian(X[:,j]) if not np.isnan(np.nanmedian(X[:,j])) else 0
    return X

class LogGBM:
    def __init__(self,rs=42):
        self.g=GradientBoostingRegressor(n_estimators=N_TREES,learning_rate=LR,
            max_depth=MAX_DEPTH,min_samples_leaf=MIN_LEAF,subsample=SUBSAMPLE,random_state=rs)
    def fit(self,X,y,**kw): self.g.fit(_imp(X),np.log1p(np.maximum(y,0)));return self
    def predict(self,X): return np.expm1(self.g.predict(_imp(X)))

class RidgeLog:
    def __init__(self,a=10.0): self.a=a
    def fit(self,X,y,**kw):
        X=_imp(X);yl=np.log1p(np.maximum(y,0));n,p=X.shape
        Xb=np.column_stack([np.ones(n),X]);I=np.eye(p+1);I[0,0]=0
        try: w=np.linalg.solve(Xb.T@Xb+self.a*I,Xb.T@yl)
        except: w,_,_,_=np.linalg.lstsq(Xb,yl,rcond=None)
        self.i=w[0];self.c=w[1:];return self
    def predict(self,X): return np.expm1(_imp(X)@self.c+self.i)

class WLag:
    def __init__(self): self.w=None;self.lc=[];self.ic=[];self.is_=1.0
    def fit(self,X,y,feature_names=None,**kw):
        fn=feature_names or []
        self.lc=[i for i,n in enumerate(fn) if 'visitor_lag' in n or 'visitor_ma' in n]
        self.ic=[i for i,n in enumerate(fn) if 'interest' in n.lower()]
        if not self.lc: self.w=np.ones(X.shape[1])/X.shape[1];return self
        corrs=[]
        for j in self.lc:
            v=~np.isnan(X[:,j])&~np.isnan(y)&(y>0)
            corrs.append(max(np.corrcoef(X[v,j],y[v])[0,1],0) if v.sum()>5 else 0)
        t=sum(corrs);self.w=np.array(corrs)/t if t>0 else np.ones(len(self.lc))/len(self.lc)
        self.is_=1.0
        if self.ic:
            lp=self._lp(X);v=~np.isnan(lp)&(y>0)&(lp>0)
            if v.sum()>5:
                r=y[v]/lp[v];iv=X[v][:,self.ic[0]];io=~np.isnan(iv)
                if io.sum()>5:
                    c=np.corrcoef(iv[io],r[io])[0,1]
                    if not np.isnan(c): self.is_=1+0.01*c
        return self
    def predict(self,X):
        p=self._lp(X)
        if self.ic and abs(self.is_-1)>0.001:
            iv=X[:,self.ic[0]];v=~np.isnan(iv);mi=np.nanmean(iv)
            if mi>0: p=p*np.clip(np.where(v,1+(iv-mi)/mi*(self.is_-1),1.0),0.8,1.2)
        return np.maximum(p,0)
    def _lp(self,X):
        X=_imp(X);r=np.zeros(len(X))
        for i,j in enumerate(self.lc):
            if i<len(self.w): r+=self.w[i]*X[:,j]
        return r

class Stacked:
    def __init__(self,models,names): self.models=models;self.names=names;self.w=None
    def fit_weights(self,Xv,yv):
        errs=[np.sqrt(np.mean((yv-np.maximum(m.predict(Xv),0))**2))+1e-6 for m in self.models]
        inv=[1/e for e in errs];t=sum(inv);self.w=[w/t for w in inv];return self
    def predict(self,X): return sum(w*np.maximum(m.predict(X),0) for m,w in zip(self.models,self.w))

def _train_stacked(Xt,yt,vf,rs=42):
    gbm=LogGBM(rs=rs);gbm.fit(Xt,yt)
    ridge=RidgeLog(RIDGE_ALPHA);ridge.fit(Xt,yt)
    wlag=WLag();wlag.fit(Xt,yt,feature_names=vf)
    models=[gbm,ridge,wlag];names=['GBM','Ridge','WLag']
    sp=max(int(len(Xt)*0.8),len(Xt)-12)
    if len(Xt)-sp>3:
        st=Stacked(models,names);st.fit_weights(Xt[sp:],yt[sp:]);return st
    return gbm

# ══════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════
script_dir=os.path.dirname(os.path.abspath(__file__))
bp=os.path.join(PROJECT_ROOT, 'data', 'country')
if not os.path.isdir(bp):
    # Fallback: search in script_dir or mnt
    bp=None
    for e in os.scandir(script_dir):
        if '방문객' in unicodedata.normalize('NFC',e.name) and e.is_dir(): bp=script_dir;break
    if not bp: print("ERROR: data not found");sys.exit(1)

print(f"Data: {bp}")
loader=V4DataLoader(bp);panel=loader.build_panel()

# ETC country
gdir=loader.global_dir;gc=None
for e in os.scandir(gdir):
    if '외래관광객 추이' in _nfc(e.name) and e.name.endswith('.csv'): gc=e.path;break
gdf=_read_csv_safe(gc);gdf.columns=[_nfc(c) for c in gdf.columns]
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
eng=V4FeatureEngineer(include_tier2=True);df_full=eng.build_features(pe)

# Imputation
MS={'멕시코','사우디아라비야','아랍에미리트','카자흐스탄','튀르키예','ETC'}
MH={'몽골','사우디아라비야','싱가포르','필리핀','홍콩','ETC'}
SF=['revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
    'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag','spend_x_fx']
HF=['hallyu_index','hallyu_x_interest']
IF_=['image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
     'image_kdrama_pct','culture_affinity','culture_x_interest']
def impute_gm(d):
    d=d.copy()
    for col in SF+HF+IF_:
        if col not in d.columns: continue
        ms=MS if col in SF else MH
        for c in ms:
            m=d['country']==c
            if not m.any(): continue
            g=d.loc[m,'country_group'].iloc[0]
            gm=(d['country_group']==g)&(~d['country'].isin(ms))
            d.loc[m,col]=d.loc[m,col].fillna(d.loc[gm,col].median())
    return d
df_imp=impute_gm(df_full)

# Feature sets
FEATS_A=[f for f in ['visitor_lag_1','visitor_lag_2','visitor_lag_3','visitor_lag_6','visitor_lag_12',
    'visitor_ma_3','visitor_ma_6','visitor_ma_12','visitor_yoy_growth','recovery_ratio',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','month_sin','month_cos'] if f in df_imp.columns]
FEATS_B=[f for f in ['exchange_rate','exchange_rate_lag_1','exchange_rate_ma_3',
    'exchange_rate_change_pct','exchange_rate_volatility','oil_price','oil_price_lag_1','oil_price_ma_3',
    'travel_interest_pct','travel_interest_lag_1','travel_interest_lag_2','travel_interest_lag_3',
    'global_travel_interest_pct','global_interest_lag_1','global_interest_lag_2','global_interest_lag_3',
    'spend_x_fx','covid_period','post_covid',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','visitor_lag_12'] if f in df_imp.columns]
FEATS_C=[f for f in ['month_sin','month_cos','quarter','is_peak_season','is_summer','is_winter',
    'weather_comfort','temp_lag_1','precip_lag_1','temp_anomaly','heavy_rain',
    'event_national_lag_1','event_seoul_lag_1','event_seoul_ratio','event_intensity','event_weighted',
    'hallyu_index','hallyu_x_interest',
    'image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct',
    'image_kdrama_pct','culture_affinity','culture_x_interest',
    'revisit_rate_lag','stay_days_lag','per_capita_spend_usd_lag','daily_spend_usd_lag',
    'overall_satisfaction_lag','revisit_intention_lag','recommend_intention_lag',
    'tourism_dominant','business_ratio','covid_period','post_covid',
    'group_A1_western_surge','group_A2_asia_surge','group_B_recovering',
    'group_C_underperform','group_ETC_group','visitor_lag_12'] if f in df_imp.columns]

print(f"Features: A={len(FEATS_A)}, B={len(FEATS_B)}, C={len(FEATS_C)}")
print(f"Data loaded in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════
#  BOOTSTRAP PREDICTION
# ══════════════════════════════════════════════════
def predict_fold_bs(df, feats, tr_mask, pr_mask):
    """Bootstrap×N: 복원추출 → N개 Stacked 학습 → 예측 평균"""
    train=df[tr_mask];pred_set=df[pr_mask];all_p=[]
    for grp in sorted(train['country_group'].unique()):
        gt=train[train['country_group']==grp];gp=pred_set[pred_set['country_group']==grp]
        if len(gp)==0: continue
        vf=[c for c in feats if c in gt.columns]
        Xt=gt[vf].values.astype(float);yt=gt['visitors'].values.astype(float)
        Xp=gp[vf].values.astype(float);yp=gp['visitors'].values.astype(float)
        meta=gp[['year_month','country','country_group']].copy()
        ok=~np.isnan(yt)&(yt>0);Xt,yt=Xt[ok],yt[ok]
        Xt=np.nan_to_num(Xt);Xp=np.nan_to_num(Xp)
        if len(Xt)<5: continue
        ok_p=~np.isnan(yp)&(yp>0)
        mv=meta.iloc[ok_p.nonzero()[0]].copy()
        Xpv=Xp[ok_p];ypv=yp[ok_p]
        if len(Xpv)==0: continue
        mx=yt.max()
        bs_preds=[]
        for b in range(N_BOOTSTRAP):
            idx=np.random.choice(len(Xt),len(Xt),replace=True)
            model=_train_stacked(Xt[idx],yt[idx],vf,rs=None)
            bs_preds.append(np.clip(model.predict(Xpv),0,mx*5))
        avg_pred=np.mean(bs_preds,axis=0)
        mv=mv.copy();mv['actual']=ypv;mv['predicted']=avg_pred;all_p.append(mv)
    return pd.concat(all_p) if all_p else pd.DataFrame()

def predict_fold_bs_nolabel(df, feats, tr_mask, pr_mask):
    """2025 예측용 — actual이 NaN/0이어도 예측 수행"""
    train=df[tr_mask];pred_set=df[pr_mask];all_p=[]
    for grp in sorted(train['country_group'].unique()):
        gt=train[train['country_group']==grp];gp=pred_set[pred_set['country_group']==grp]
        if len(gp)==0: continue
        vf=[c for c in feats if c in gt.columns]
        Xt=gt[vf].values.astype(float);yt=gt['visitors'].values.astype(float)
        Xp=gp[vf].values.astype(float);yp=gp['visitors'].values.astype(float)
        meta=gp[['year_month','country','country_group']].copy()
        ok=~np.isnan(yt)&(yt>0);Xt,yt=Xt[ok],yt[ok]
        Xt=np.nan_to_num(Xt);Xp=np.nan_to_num(Xp)
        if len(Xt)<5: continue
        mx=yt.max()
        bs_preds=[]
        for b in range(N_BOOTSTRAP):
            idx=np.random.choice(len(Xt),len(Xt),replace=True)
            model=_train_stacked(Xt[idx],yt[idx],vf,rs=None)
            bs_preds.append(np.clip(model.predict(Xp),0,mx*5))
        avg_pred=np.mean(bs_preds,axis=0)
        mv=meta.copy();mv['actual']=yp;mv['predicted']=avg_pred;all_p.append(mv)
    return pd.concat(all_p) if all_p else pd.DataFrame()

# ══════════════════════════════════════════════════
#  1. OOF (4-fold Rolling Window)
# ══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  1. OOF: Bootstrap×{N_BOOTSTRAP} (4-fold Rolling Window)")
print(f"{'='*70}")

folds=[
    ('Fold1: 2018→2019H1',
     lambda y:'2018-01'<=y<='2018-12',lambda y:'2019-01'<=y<='2019-06'),
    ('Fold2: ~2019H1→2019H2',
     lambda y:'2018-01'<=y<='2019-06',lambda y:'2019-07'<=y<='2019-12'),
    ('Fold3: ~2019+2023H2→2024H1',
     lambda y:('2018-01'<=y<='2019-12')or('2023-07'<=y<='2023-12'),
     lambda y:'2024-01'<=y<='2024-06'),
    ('Fold4: ~2024H1→2024H2',
     lambda y:('2018-01'<=y<='2019-12')or('2023-07'<=y<='2024-06'),
     lambda y:'2024-07'<=y<='2024-12'),
]

oof={k:[] for k in 'ABC'}
for nm,tfn,pfn in folds:
    tr=df_imp['year_month'].apply(tfn).values
    pr=df_imp['year_month'].apply(pfn).values
    print(f"\n  {nm}: train={tr.sum()}, pred={pr.sum()}")
    for lbl,feats in [('A',FEATS_A),('B',FEATS_B),('C',FEATS_C)]:
        t1=time.time()
        po=predict_fold_bs(df_imp,feats,tr,pr)
        dt=time.time()-t1
        if not po.empty:
            oof[lbl].append(po)
            m=np.mean(np.abs((po['actual']-po['predicted'])/po['actual']))*100
            print(f"    {lbl}: MAPE={m:.1f}% [{dt:.1f}s]")

for k in oof:
    oof[k]=pd.concat(oof[k]) if oof[k] else pd.DataFrame()

# ══════════════════════════════════════════════════
#  2. META-LEARNER: BayesianRidge(log) on OOF
# ══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  2. META-LEARNER: BayesianRidge(log)")
print(f"{'='*70}")

def merge_oof(d):
    mg=d['A'][['year_month','country','actual','predicted']].rename(columns={'predicted':'pred_A'})
    mg=mg.merge(d['B'][['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),on=['year_month','country'],how='inner')
    mg=mg.merge(d['C'][['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),on=['year_month','country'],how='inner')
    return mg

mg=merge_oof(oof)
X_oof=mg[['pred_A','pred_B','pred_C']].values;y_oof=mg['actual'].values
Xl=np.log1p(np.maximum(X_oof,0));yl=np.log1p(np.maximum(y_oof,0))
br_meta=SkBayesianRidge(max_iter=300,fit_intercept=True)
br_meta.fit(Xl,yl)
oof_final=np.maximum(np.expm1(br_meta.predict(Xl)),0)

print(f"  BayesRidge coef: A={br_meta.coef_[0]:.4f}, B={br_meta.coef_[1]:.4f}, C={br_meta.coef_[2]:.4f}")
print(f"  BayesRidge intercept: {br_meta.intercept_:.4f}")
print(f"  Alpha: {br_meta.alpha_:.4f}, Lambda: {br_meta.lambda_:.4f}")

# Error correlation
ea=X_oof[:,0]-y_oof;eb=X_oof[:,1]-y_oof;ec=X_oof[:,2]-y_oof
ecAB=np.corrcoef(ea,eb)[0,1];ecAC=np.corrcoef(ea,ec)[0,1];ecBC=np.corrcoef(eb,ec)[0,1]
print(f"\n  Error Correlations: AB={ecAB:.3f}, AC={ecAC:.3f}, BC={ecBC:.3f}")
print(f"  Avg |Error Corr|: {(abs(ecAB)+abs(ecAC)+abs(ecBC))/3:.3f}")

# OOF Performance
def ev(name,p,a):
    r=np.sqrt(np.mean((a-p)**2));m=np.mean(np.abs((a-p)/a))*100;te=abs(a.sum()-p.sum())/a.sum()*100
    print(f"  {name:<28s} RMSE={r:>10,.0f}  MAPE={m:>6.1f}%  TotalErr={te:>5.1f}%")
    return m

print(f"\n  OOF Performance:")
ev("Model A (Lag+BS)",X_oof[:,0],y_oof)
ev("Model B (Macro+BS)",X_oof[:,1],y_oof)
ev("Model C (Event+BS)",X_oof[:,2],y_oof)
oof_mape=ev("★ BS+BayesRidge(log)",oof_final,y_oof)

# ══════════════════════════════════════════════════
#  3. OOF 국가별 성능
# ══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  3. OOF 국가별 MAPE")
print(f"{'='*70}")

mg['pred_final']=oof_final
print(f"\n  {'Country':<14s} {'MAPE':>7s} {'RMSE':>10s} {'N':>4s}")
print(f"  {'-'*38}")
c_perf=[]
for c in sorted(mg['country'].unique()):
    cm=mg[mg['country']==c]
    m=np.mean(np.abs((cm['actual']-cm['pred_final'])/cm['actual']))*100
    r=np.sqrt(np.mean((cm['actual']-cm['pred_final'])**2))
    c_perf.append((c,m,r,len(cm)))
c_perf.sort(key=lambda x:x[1])
for c,m,r,n in c_perf:
    print(f"  {c:<14s} {m:>6.1f}% {r:>10,.0f} {n:>4d}")
print(f"  {'─'*38}")
print(f"  {'전체':<14s} {oof_mape:>6.1f}%")

# ══════════════════════════════════════════════════
#  4. 2024 TEST (전체 학습 → 2024 예측)
# ══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  4. 2024 TEST")
print(f"{'='*70}")

is_covid=lambda ym:'2020-02'<=ym<='2023-06'
test_tr=df_imp['year_month'].apply(lambda y:y<='2023-12' and not is_covid(y) and y>='2018-01').values
test_pr=df_imp['year_month'].apply(lambda y:'2024-01'<=y<='2024-12').values
print(f"  Train: {test_tr.sum()}, Pred: {test_pr.sum()}")

test_preds={}
for lbl,feats in [('A',FEATS_A),('B',FEATS_B),('C',FEATS_C)]:
    test_preds[lbl]=predict_fold_bs(df_imp,feats,test_tr,test_pr)

# Merge & apply meta
tA=test_preds['A'];tB=test_preds['B'];tC=test_preds['C']
if not tA.empty and not tB.empty and not tC.empty:
    tm=tA[['year_month','country','country_group','actual','predicted']].rename(columns={'predicted':'pred_A'})
    tm=tm.merge(tB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),on=['year_month','country'],how='inner')
    tm=tm.merge(tC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),on=['year_month','country'],how='inner')
    Xt=np.log1p(np.maximum(tm[['pred_A','pred_B','pred_C']].values,0))
    tm['pred_final']=np.maximum(np.expm1(br_meta.predict(Xt)),0)

    a=tm['actual'].values;p=tm['pred_final'].values
    test_mape=np.mean(np.abs((a-p)/a))*100
    test_rmse=np.sqrt(np.mean((a-p)**2))
    print(f"\n  2024 Test: MAPE={test_mape:.1f}%, RMSE={test_rmse:,.0f}")

    # 월별
    monthly=tm.groupby('year_month').agg({'actual':'sum','pred_final':'sum'}).reset_index().sort_values('year_month')
    print(f"\n  {'Month':<10s} {'Actual':>12s} {'Predicted':>12s} {'Error':>8s}")
    print(f"  {'-'*44}")
    for _,r in monthly.iterrows():
        e=(r['pred_final']-r['actual'])/r['actual']*100
        print(f"  {r['year_month']:<10s} {r['actual']:>12,.0f} {r['pred_final']:>12,.0f} {e:>+7.1f}%")

# ══════════════════════════════════════════════════
#  5. 2025 PREDICTION
# ══════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  5. 2025 PREDICTION (Final)")
print(f"{'='*70}")

final_tr=df_imp['year_month'].apply(lambda y:y<='2024-12' and not is_covid(y) and y>='2018-01').values
final_pr=df_imp['year_month'].apply(lambda y:'2025-01'<=y<='2025-12').values
print(f"  Train: {final_tr.sum()}, Pred: {final_pr.sum()}")

final_preds={}
for lbl,feats in [('A',FEATS_A),('B',FEATS_B),('C',FEATS_C)]:
    final_preds[lbl]=predict_fold_bs_nolabel(df_imp,feats,final_tr,final_pr)

fA=final_preds['A'];fB=final_preds['B'];fC=final_preds['C']
if not fA.empty and not fB.empty and not fC.empty:
    fm=fA[['year_month','country','country_group','actual','predicted']].rename(columns={'predicted':'pred_A'})
    fm=fm.merge(fB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),on=['year_month','country'],how='inner')
    fm=fm.merge(fC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),on=['year_month','country'],how='inner')
    Xf=np.log1p(np.maximum(fm[['pred_A','pred_B','pred_C']].values,0))
    fm['pred_final']=np.maximum(np.expm1(br_meta.predict(Xf)),0)

    # 실제값 있는 월 확인
    has_actual=(fm['actual'].notna())&(fm['actual']>0)

    # 월별 합계
    monthly=fm.groupby('year_month').agg({
        'pred_A':'sum','pred_B':'sum','pred_C':'sum','pred_final':'sum','actual':'sum'
    }).reset_index().sort_values('year_month')

    print(f"\n  {'Month':<10s} {'Predicted':>12s} {'Actual':>12s} {'Error':>8s}")
    print(f"  {'-'*44}")
    for _,r in monthly.iterrows():
        a_str=f"{r['actual']:>12,.0f}" if r['actual']>0 else f"{'─':>12s}"
        e_str=f"{(r['pred_final']-r['actual'])/r['actual']*100:>+7.1f}%" if r['actual']>0 else f"{'─':>8s}"
        print(f"  {r['year_month']:<10s} {r['pred_final']:>12,.0f} {a_str} {e_str}")
    total_pred=monthly['pred_final'].sum()
    total_act=monthly['actual'].sum()
    e_total=f"{(total_pred-total_act)/total_act*100:>+7.1f}%" if total_act>0 else "─"
    print(f"  {'TOTAL':<10s} {total_pred:>12,.0f} {total_act:>12,.0f} {e_total:>8s}")

    # 국가별 연간 합계
    print(f"\n  {'Country':<14s} {'2025 예측':>12s} {'실제값':>12s}")
    print(f"  {'-'*40}")
    country_yr=fm.groupby('country').agg({'pred_final':'sum','actual':'sum'}).reset_index()
    country_yr=country_yr.sort_values('pred_final',ascending=False)
    for _,r in country_yr.iterrows():
        a_str=f"{r['actual']:>12,.0f}" if r['actual']>0 else f"{'─':>12s}"
        print(f"  {r['country']:<14s} {r['pred_final']:>12,.0f} {a_str}")
    print(f"  {'─'*40}")
    print(f"  {'합계':<14s} {country_yr['pred_final'].sum():>12,.0f}")

    # CSV 저장
    out_dir=os.path.join(PROJECT_ROOT, 'outputs')
    fm.to_csv(os.path.join(out_dir,'v9_final_2025_predictions.csv'),index=False,encoding='utf-8-sig')
    if not tm.empty:
        tm.to_csv(os.path.join(out_dir,'v9_final_2024_test.csv'),index=False,encoding='utf-8-sig')
    print(f"\n  💾 저장: v9_final_2025_predictions.csv, v9_final_2024_test.csv")

print(f"\n  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*70}")
print("  V9 Final DONE")
print(f"{'='*70}")



# ══════════════════════════════════════════════════
#  6. 2026 FORECAST (Dampened Rolling — 월별 순차 예측)
# ══════════════════════════════════════════════════
DAMP_ALPHA = 0.7  # 예측값 비중 (0.7 = 예측 70% + 작년실제 30%)

print(f"\n{'='*70}")
print(f"  6. 2026 FORECAST (Dampened Rolling, alpha={DAMP_ALPHA})")
print(f"  Training: 2018-2025 (COVID excluded)")
print(f"  Predicting: 2026-01 ~ 2026-12 (month by month)")
print(f"  Dampening: lag = {DAMP_ALPHA}×pred + {1-DAMP_ALPHA:.1f}×last_year_actual")
print(f"{'='*70}")

# 2025 실제값 lookup (dampening 용)
actual_2025 = {}
for _, r in df_imp[df_imp['year_month'].str[:4]=='2025'].iterrows():
    actual_2025[(r['country'], r['month'])] = r['visitors']

# 2026 placeholder 추가
rows_2026=[]
for country in sorted(df_imp['country'].unique()):
    cd=df_imp[df_imp['country']==country]
    for month in range(1,13):
        same_m=cd[(cd['year']==2025)&(cd['month']==month)]
        if len(same_m)>0:
            row=same_m.iloc[0].to_dict()
        else:
            row={'country':country,'country_group':cd['country_group'].iloc[0]}
        row['year_month']=f'2026-{month:02d}'
        row['year']=2026
        row['month']=month
        row['visitors']=np.nan
        rows_2026.append(row)

df_2026=pd.DataFrame(rows_2026)
df_ext=pd.concat([df_imp,df_2026],ignore_index=True).sort_values(['country','year_month']).reset_index(drop=True)

def recalc_lags(df):
    """Lag/MA 피처 재계산"""
    for lag in [1,2,3,6,12]:
        col=f'visitor_lag_{lag}'
        if col in df.columns:
            df[col]=df.groupby('country')['visitors'].shift(lag)
    for w in [3,6,12]:
        col=f'visitor_ma_{w}'
        if col in df.columns:
            df[col]=df.groupby('country')['visitors'].transform(
                lambda x:x.shift(1).rolling(w,min_periods=1).mean())
    if 'visitor_yoy_growth' in df.columns:
        df['visitor_yoy_growth']=df.groupby('country').apply(
            lambda g:(g['visitors'].shift(1)/g['visitors'].shift(13)-1)*100
        ).reset_index(level=0,drop=True)
    if 'recovery_ratio' in df.columns:
        b19=df[df['year']==2019][['country','month','visitors']].rename(columns={'visitors':'b19'})
        if not b19.empty:
            df=df.drop(columns=['recovery_ratio'],errors='ignore')
            df=df.merge(b19,on=['country','month'],how='left')
            df['recovery_ratio']=np.where(df['b19']>0,df['visitors']/df['b19'],np.nan)
            df['recovery_ratio']=df.groupby('country')['recovery_ratio'].shift(1)
            df.drop(columns=['b19'],errors='ignore',inplace=True)
    return df

# Dampened Rolling: 예측 → dampened값을 visitors에 넣기 → 다음 달 lag 계산
f26_results=[]
for pred_month in range(1,13):
    pred_ym=f'2026-{pred_month:02d}'
    
    # Lag 재계산
    df_ext=recalc_lags(df_ext)
    
    # Train/Pred mask
    f26_tr=df_ext['year_month'].apply(lambda y:y<='2025-12' and not is_covid(y) and y>='2018-01').values
    f26_pr=(df_ext['year_month']==pred_ym).values
    
    # 3개 모델 예측
    month_preds={}
    for lbl,feats in [('A',FEATS_A),('B',FEATS_B),('C',FEATS_C)]:
        month_preds[lbl]=predict_fold_bs_nolabel(df_ext,feats,f26_tr,f26_pr)
    
    mA,mB,mC=month_preds['A'],month_preds['B'],month_preds['C']
    if not mA.empty and not mB.empty and not mC.empty:
        mm=mA[['year_month','country','country_group','predicted']].rename(columns={'predicted':'pred_A'})
        mm=mm.merge(mB[['year_month','country','predicted']].rename(columns={'predicted':'pred_B'}),on=['year_month','country'],how='inner')
        mm=mm.merge(mC[['year_month','country','predicted']].rename(columns={'predicted':'pred_C'}),on=['year_month','country'],how='inner')
        Xm=np.log1p(np.maximum(mm[['pred_A','pred_B','pred_C']].values,0))
        mm['pred_2026']=np.maximum(np.expm1(br_meta.predict(Xm)),0)
        
        # Dampened 값을 visitors에 채워넣기 (다음 달 lag용)
        # pred_2026은 원래 예측값 그대로 저장, lag에만 dampened 적용
        for _,row in mm.iterrows():
            mask=(df_ext['year_month']==pred_ym)&(df_ext['country']==row['country'])
            pred_val = row['pred_2026']
            # 작년 동월 실제값
            ly_actual = actual_2025.get((row['country'], pred_month), pred_val)
            # Dampened value: 예측×alpha + 작년실제×(1-alpha)
            dampened = DAMP_ALPHA * pred_val + (1 - DAMP_ALPHA) * ly_actual
            df_ext.loc[mask,'visitors'] = dampened
        
        f26_results.append(mm)
        total_pred=mm['pred_2026'].sum()
        print(f"  {pred_ym}: {total_pred:>12,.0f} ({len(mm)} countries)")

f26m=pd.concat(f26_results,ignore_index=True)

# 2024/2025 실적 병합
f26m['month_key']=f26m['year_month'].str[-2:]
for yr_label in ['2024','2025']:
    tmp=df_imp[df_imp['year_month'].str[:4]==yr_label][['country','month','visitors']].copy()
    tmp['month_key']=tmp['month'].astype(str).str.zfill(2)
    tmp=tmp.rename(columns={'visitors':f'actual_{yr_label}'}).drop(columns=['month'])
    f26m=f26m.merge(tmp,on=['country','month_key'],how='left')

# 국가별 합계
print(f"\n  {'국가':<14s} {'2024':>12s} {'2025':>12s} {'2026F':>12s} {'25→26':>8s}")
print(f"  {'─'*60}")
for c in sorted(f26m['country'].unique()):
    cf=f26m[f26m['country']==c]
    c24,c25,c26=cf['actual_2024'].sum(),cf['actual_2025'].sum(),cf['pred_2026'].sum()
    g=(c26/c25-1)*100 if c25>0 else 0
    print(f"  {c:<14s} {c24:>12,.0f} {c25:>12,.0f} {c26:>12,.0f} {g:>+7.1f}%")
t24,t25,t26=f26m['actual_2024'].sum(),f26m['actual_2025'].sum(),f26m['pred_2026'].sum()
print(f"  {'─'*60}")
print(f"  {'합계':<14s} {t24:>12,.0f} {t25:>12,.0f} {t26:>12,.0f} {(t26/t25-1)*100:>+7.1f}%")

# 월별 합계
print(f"\n  {'월':>5s} {'2024':>12s} {'2025':>12s} {'2026F':>12s} {'25→26':>8s}")
print(f"  {'─'*50}")
for m in range(1,13):
    mf=f26m[f26m['year_month']==f'2026-{m:02d}']
    m24,m25,m26=mf['actual_2024'].sum(),mf['actual_2025'].sum(),mf['pred_2026'].sum()
    print(f"  {m:>3d}월  {m24:>12,.0f} {m25:>12,.0f} {m26:>12,.0f} {(m26/m25-1)*100 if m25>0 else 0:>+7.1f}%")

# CSV 저장
out26=f26m[['year_month','country','country_group','actual_2024','actual_2025','pred_A','pred_B','pred_C','pred_2026']].copy()
out26['pred_2026']=out26['pred_2026'].round(0).astype(int)
out26.to_csv(os.path.join(PROJECT_ROOT,'outputs','v9_forecast_2026_dampened.csv'),index=False,encoding='utf-8-sig')
print(f"\n  💾 v9_forecast_2026_dampened.csv 저장 완료")

print(f"\n  총 소요시간 (2026 포함): {time.time()-t0:.0f}s")
print(f"{'='*70}")
print("  V9 Final + 2026 Dampened Forecast DONE")
print(f"{'='*70}")