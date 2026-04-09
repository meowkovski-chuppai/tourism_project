"""
v4_engineer.py — V4 피처 엔지니어링 (확장)
V3 features + 날씨, 이벤트, 만족도, 이미지, 목적 피처
"""
import pandas as pd
import numpy as np
from typing import List, Dict

class V4FeatureEngineer:
    LAG_MONTHS = [1, 2, 3, 6, 12]
    MA_WINDOWS = [3, 6, 12]

    def __init__(self, include_tier2=True):
        self.include_tier2 = include_tier2

    def build_features(self, panel):
        df = panel.copy().sort_values(['country','year_month']).reset_index(drop=True)
        print("Building V4 features...")
        df = self._build_core(df)
        df = self._build_weather(df)
        df = self._build_events(df)
        df = self._build_satisfaction(df)
        df = self._build_image(df)
        df = self._build_purpose(df)
        if self.include_tier2:
            df = self._build_tier2(df)
        feat_cols = self.get_feature_columns(df)
        print(f"✅ V4 Features: {len(feat_cols)} feature columns")
        return df

    def _build_core(self, df):
        """V3 core features: time, lags, exchange rate, oil, interest, covid"""
        df['month'] = df['year_month'].str[5:7].astype(int)
        df['quarter'] = (df['month']-1)//3+1
        df['year'] = df['year_month'].str[:4].astype(int)
        df['is_peak_season'] = df['month'].isin([4,7,8,10]).astype(int)
        df['is_summer'] = df['month'].isin([6,7,8]).astype(int)
        df['is_winter'] = df['month'].isin([12,1,2]).astype(int)
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)

        # Visitor lags
        for lag in self.LAG_MONTHS:
            df[f'visitor_lag_{lag}'] = df.groupby('country')['visitors'].shift(lag)
        for w in self.MA_WINDOWS:
            df[f'visitor_ma_{w}'] = df.groupby('country')['visitors'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df['visitor_yoy_growth'] = df.groupby('country').apply(
            lambda g: (g['visitors'].shift(1)/g['visitors'].shift(13)-1)*100
        ).reset_index(level=0, drop=True)

        # Exchange rate
        if 'exchange_rate' in df.columns:
            df['exchange_rate_lag_1'] = df.groupby('country')['exchange_rate'].shift(1)
            df['exchange_rate_ma_3'] = df.groupby('country')['exchange_rate'].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            df['exchange_rate_change_pct'] = df.groupby('country').apply(
                lambda g: (g['exchange_rate']/g['exchange_rate'].shift(1)-1)*100
            ).reset_index(level=0, drop=True)
            # NEW V4: exchange rate volatility (rolling std)
            df['exchange_rate_volatility'] = df.groupby('country')['exchange_rate'].transform(
                lambda x: x.shift(1).rolling(6, min_periods=3).std())

        # Oil price
        if 'oil_price' in df.columns:
            df['oil_price_lag_1'] = df.groupby('country')['oil_price'].shift(1)
            df['oil_price_ma_3'] = df.groupby('country')['oil_price'].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())

        # Travel interest
        if 'travel_interest_pct' in df.columns:
            for lag in [1,2,3]:
                df[f'travel_interest_lag_{lag}'] = df.groupby('country')['travel_interest_pct'].shift(lag)
        if 'global_travel_interest_pct' in df.columns:
            for lag in [1,2,3]:
                df[f'global_interest_lag_{lag}'] = df.groupby('country')['global_travel_interest_pct'].shift(lag)

        # COVID
        df['covid_period'] = ((df['year_month']>='2020-02')&(df['year_month']<='2022-06')).astype(int)
        df['post_covid'] = (df['year_month']>='2022-07').astype(int)
        df = self._add_recovery_ratio(df)

        # Group dummies
        if 'country_group' in df.columns:
            dummies = pd.get_dummies(df['country_group'], prefix='group', dtype=int)
            df = pd.concat([df, dummies], axis=1)

        return df

    def _build_weather(self, df):
        """날씨 피처: 기온, 강수량, 기온쾌적도"""
        if 'seoul_temp' in df.columns:
            # Lag weather (previous month)
            # Weather is same for all countries at given month, no need for groupby
            df['temp_lag_1'] = df.groupby('country')['seoul_temp'].shift(1)
            df['precip_lag_1'] = df.groupby('country')['seoul_precip'].shift(1)
            
            # Weather comfort index: 15~25도 = 1 (comfortable), 극단 = 0
            df['weather_comfort'] = 1 - np.minimum(
                np.abs(df['seoul_temp'].fillna(15) - 20) / 25, 1)
            
            # Temperature anomaly: deviation from monthly average
            monthly_avg = df.groupby('month')['seoul_temp'].transform('mean')
            df['temp_anomaly'] = df['seoul_temp'] - monthly_avg
            
            # Heavy rain flag
            df['heavy_rain'] = (df['seoul_precip'].fillna(0) > 200).astype(int)
        
        return df

    def _build_events(self, df):
        """이벤트 피처: 축제 수, 서울 비율, 이벤트 강도"""
        if 'event_count_national' in df.columns:
            # Lag events
            df['event_national_lag_1'] = df.groupby('country')['event_count_national'].shift(1)
            df['event_seoul_lag_1'] = df.groupby('country')['event_count_seoul'].shift(1)
            
            # Seoul event ratio
            df['event_seoul_ratio'] = np.where(
                df['event_count_national'] > 0,
                df['event_count_seoul'] / df['event_count_national'],
                0
            )
            
            # Event intensity (log-transformed)
            df['event_intensity'] = np.log1p(df['event_count_national'].fillna(0))
            
            # Country-weighted events (approximate by group)
            # Western/surge tourists: Seoul-heavy (70%), B_recovering: mixed, etc.
            weight_map = {
                'A1_western_surge': 0.7,  # Mostly Seoul
                'A2_asia_surge': 0.5,     # Seoul + regions
                'B_recovering': 0.4,       # Japan/China spread more (Jeju, etc.)
                'C_underperform': 0.6,
            }
            df['event_weighted'] = df.apply(
                lambda r: (r.get('event_count_seoul',0) * weight_map.get(r.get('country_group',''),0.5) +
                          r.get('event_count_national',0) * (1-weight_map.get(r.get('country_group',''),0.5))),
                axis=1
            )
        
        return df

    def _build_satisfaction(self, df):
        """만족도 피처: 재방문율, 지출, 만족도 (연도별 → 이미 merge됨)"""
        # These are already merged at annual level in the panel
        # Forward-fill within country (annual data fills all months of that year)
        sat_cols = ['revisit_rate','stay_days','per_capita_spend_usd','daily_spend_usd',
                   'overall_satisfaction','revisit_intention','recommend_intention']
        for col in sat_cols:
            if col in df.columns:
                # Create lag version (previous year's value) to avoid leakage
                df[f'{col}_lag'] = df.groupby('country')[col].shift(12)
                # Also keep the direct value for countries where we have it
        
        # Spending × exchange rate interaction
        if 'per_capita_spend_usd' in df.columns and 'exchange_rate_change_pct' in df.columns:
            df['spend_x_fx'] = (
                df['per_capita_spend_usd'].fillna(0) * 
                df['exchange_rate_change_pct'].fillna(0) / 100
            )
        
        return df

    def _build_image(self, df):
        """한국 이미지 피처 (country-level, static)"""
        # These are already merged as country-level constants
        # Create interaction terms
        img_cols = ['image_kfood_pct','image_kpop_pct','image_kbeauty_pct','image_kdrama_pct']
        present_cols = [c for c in img_cols if c in df.columns]
        
        if present_cols:
            # Culture affinity score: sum of all cultural image percentages
            df['culture_affinity'] = df[present_cols].fillna(0).sum(axis=1)
            
            # K-content × travel interest interaction
            if 'travel_interest_pct' in df.columns:
                df['culture_x_interest'] = (
                    df['culture_affinity'] * df['travel_interest_pct'].fillna(0) / 100
                )
        
        return df

    def _build_purpose(self, df):
        """방문 목적 피처"""
        if 'tourism_pct' in df.columns:
            # Tourism-dominant flag
            df['tourism_dominant'] = (df['tourism_pct'].fillna(0) > 80).astype(int)
            
            # Business travel indicator
            if 'business_pct' in df.columns:
                df['business_ratio'] = df['business_pct'].fillna(0)
        
        return df

    def _build_tier2(self, df):
        """Tier 2: hallyu × interest interaction"""
        if 'hallyu_index' in df.columns and 'travel_interest_pct' in df.columns:
            df['hallyu_x_interest'] = (
                df['hallyu_index'] * df['travel_interest_pct'].fillna(0) / 100)
        return df

    def _add_recovery_ratio(self, df):
        baseline_2019 = (
            df[df['year']==2019][['country','month','visitors']]
            .rename(columns={'visitors':'baseline_2019'})
        )
        if baseline_2019.empty:
            df['recovery_ratio'] = np.nan
            return df
        df = df.merge(baseline_2019, on=['country','month'], how='left')
        df['recovery_ratio'] = np.where(
            df['baseline_2019']>0, df['visitors']/df['baseline_2019'], np.nan)
        df['recovery_ratio'] = df.groupby('country')['recovery_ratio'].shift(1)
        df.drop(columns=['baseline_2019'], errors='ignore', inplace=True)
        return df

    def get_feature_columns(self, df, tier=2):
        exclude = {
            'year_month','country','country_group','visitors','year','month',
            'baseline_2019',
            # Raw satisfaction (use lagged versions)
            'revisit_rate','stay_days','per_capita_spend_usd','daily_spend_usd',
            'overall_satisfaction','revisit_intention','recommend_intention',
            # Raw weather/event (use processed versions)
            'seoul_temp','seoul_precip','nat_temp','nat_precip',
            'event_count_national','event_count_seoul',
        }
        tier2_cols = {'hallyu_index','hallyu_x_interest'}
        
        feature_cols = []
        for col in df.columns:
            if col in exclude: continue
            if tier == 1 and col in tier2_cols: continue
            if df[col].dtype in ['float64','int64','int32','float32','uint8']:
                feature_cols.append(col)
        return feature_cols

    def split_train_test(self, df, train_end='2024-06', test_start='2024-07',
                         test_end='2024-12', predict_start='2025-01', exclude_covid=True):
        train = df[df['year_month']<=train_end].copy()
        test = df[(df['year_month']>=test_start)&(df['year_month']<=test_end)].copy()
        predict = df[df['year_month']>=predict_start].copy()
        if exclude_covid:
            covid = (train['year_month']>='2020-02')&(train['year_month']<='2022-06')
            train = train[~covid].copy()
        print(f"Train: {len(train)} rows ({train['year_month'].min()}~{train['year_month'].max()})")
        if exclude_covid: print(f"  (COVID 2020-02~2022-06 excluded)")
        print(f"Test: {len(test)} rows, Predict: {len(predict)} rows")
        return {'train':train,'test':test,'predict':predict}
