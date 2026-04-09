"""
v4_loader.py — V4 국가별 데이터 로더 (확장)
V3 loader + 날씨, 이벤트, 한국이미지, 방문목적 추가 로딩
"""
import os, re, unicodedata, glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def _nfc(s): return unicodedata.normalize('NFC', s)

# V3 4-group classification (recovery-rate based)
COUNTRY_GROUPS = {
    'A1_western_surge': [
        '미국','영국','프랑스','독일','캐나다','호주',
        '멕시코','튀르키예','사우디아라비야','카자흐스탄',
    ],
    'A2_asia_surge': ['싱가포르','대만','인도','인도네시아','필리핀','몽골'],
    'B_recovering': ['일본','중국','베트남','홍콩','말레이시아'],
    'C_underperform': ['러시아','태국','아랍에미리트'],
}

HALLYU_MISSING = {'몽골','사우디아라비야','싱가포르','필리핀','홍콩'}

def get_country_group(country):
    for g, cs in COUNTRY_GROUPS.items():
        if country in cs:
            return g
    return 'other'

def _read_csv_safe(path, **kw):
    for enc in ['utf-8','utf-8-sig','euc-kr','cp949']:
        try: return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, UnicodeError): continue
    return None

def _normalize_yearmonth(val):
    val = str(val).strip()
    m = re.match(r'(\d{4})년?\s*(\d{1,2})월?', val)
    if m: return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r'(\d{4})[./\-](\d{1,2})', val)
    if m: return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.match(r'^(\d{4})(\d{2})$', val)
    if m: return f"{m.group(1)}-{m.group(2)}"
    return None

def _find_country_dirs(base_path):
    dirs = {}
    for entry in os.scandir(base_path):
        name = _nfc(entry.name)
        if entry.is_dir() and '_방문객' in name and '글로벌' not in name:
            dirs[name.replace('_방문객','')] = entry.path
        elif entry.is_dir() and '글로벌' in name:
            dirs['__global__'] = entry.path
    return dirs

def _find_file(directory, keyword):
    for entry in os.scandir(directory):
        name_nfc = _nfc(entry.name)
        if entry.is_file() and keyword in name_nfc and name_nfc.endswith('.csv'):
            return entry.path
    return None

def _find_file_xlsx(directory, keyword):
    for entry in os.scandir(directory):
        name_nfc = _nfc(entry.name)
        if entry.is_file() and keyword in name_nfc and name_nfc.endswith('.xlsx'):
            return entry.path
    return None


class V4DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.country_dirs = _find_country_dirs(base_path)
        self.countries = [k for k in self.country_dirs if k != '__global__']
        self.global_dir = self.country_dirs.get('__global__')
        # project root: base_path is data/country/, so project root is ../../
        self.project_path = os.path.dirname(os.path.dirname(base_path))

    # ── Core: monthly visitors (same as V3) ──
    def load_monthly_visitors(self):
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '방한 외래관광객 추이')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df = df.rename(columns={
                '기준연월':'year_month','방한 외래관광객(명)':'visitors',
                '환율(원)':'exchange_rate','유가(달러)':'oil_price',
            })
            if 'year_month' in df.columns:
                df['year_month'] = df['year_month'].astype(str).apply(_normalize_yearmonth)
                df = df.dropna(subset=['year_month'])
            df['country'] = country
            cols = ['year_month','country']
            for c in ['visitors','exchange_rate','oil_price']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
                    cols.append(c)
            frames.append(df[cols])
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(['country','year_month']).reset_index(drop=True)

    # ── Travel interest ──
    def load_travel_interest(self):
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '관심도 추이')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df = df.rename(columns={'기준연월':'year_month','값(%)':'travel_interest_pct'})
            if 'year_month' in df.columns:
                df['year_month'] = df['year_month'].astype(str).apply(_normalize_yearmonth)
            if 'travel_interest_pct' in df.columns:
                df['travel_interest_pct'] = pd.to_numeric(
                    df['travel_interest_pct'].astype(str).str.replace(',',''), errors='coerce')
            df['country'] = country
            frames.append(df[['year_month','country','travel_interest_pct']].dropna())
        # Global
        if self.global_dir:
            fpath = _find_file(self.global_dir, '관심도 추이')
            if fpath:
                df = _read_csv_safe(fpath)
                if df is not None:
                    df = df.rename(columns={'기준연월':'year_month','값(%)':'travel_interest_pct'})
                    if 'year_month' in df.columns:
                        df['year_month'] = df['year_month'].astype(str).apply(_normalize_yearmonth)
                    if 'travel_interest_pct' in df.columns:
                        df['travel_interest_pct'] = pd.to_numeric(
                            df['travel_interest_pct'].astype(str).str.replace(',',''), errors='coerce')
                    df['country'] = '__global__'
                    frames.append(df[['year_month','country','travel_interest_pct']].dropna())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── Hallyu interest ──
    def load_hallyu_interest(self):
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '한류 관심도')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df['country'] = country
            frames.append(df)
        if not frames: return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        # Summarize to country-level score
        if '비율' in result.columns:
            result['비율'] = pd.to_numeric(result['비율'].astype(str).str.replace(',',''), errors='coerce')
            if '기준연도' in result.columns:
                result['기준연도'] = pd.to_numeric(result['기준연도'], errors='coerce')
                recent = result[result['기준연도'] >= result['기준연도'].max() - 2]
            else:
                recent = result
            summary = recent.groupby('country')['비율'].mean().reset_index()
            summary = summary.rename(columns={'비율':'hallyu_index'})
            return summary
        return pd.DataFrame()

    # ── Annual satisfaction ──
    def load_annual_satisfaction(self):
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '행태')
            if not fpath: fpath = _find_file(dirpath, '만족도')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df.columns = [_nfc(c) for c in df.columns]
            rename = {
                '기준년도':'year','재방문율(%)':'revisit_rate',
                '체재 기간(일)':'stay_days',
                '1인 평균 지출 경비(USD)':'per_capita_spend_usd',
                '1인 평균 지출 경비(USS)':'per_capita_spend_usd',
                '1일 평균 지출 경비(USD)':'daily_spend_usd',
                '1일 평균 지출 경비(USS)':'daily_spend_usd',
                '전반적 만족도(%)':'overall_satisfaction',
                '전반적 만족도(긍정 응답 비율)':'overall_satisfaction',
                '재방문 의향(%)':'revisit_intention',
                '관광목적 재방문 의향(긍정 응답 비율)':'revisit_intention',
                '타인 추천 의향(%)':'recommend_intention',
                '타인 추천 의향(긍정 응답 비율)':'recommend_intention',
            }
            df = df.rename(columns=rename)
            df['country'] = country
            for col in ['year','revisit_rate','stay_days','per_capita_spend_usd',
                        'daily_spend_usd','overall_satisfaction','revisit_intention','recommend_intention']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',',''), errors='coerce')
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── NEW: Korea Image ──
    def load_korea_image(self):
        """국가별 한국연상이미지 → 국가별 top-1 이미지 + K-Food/K-Pop 비율"""
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '한국연상이미지')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df.columns = [_nfc(c) for c in df.columns]
            df['country'] = country
            # Parse: 순위, 기준연도, 연상이미지(%), 기준국가명
            if '연상이미지(%)' in df.columns and '순위' in df.columns and '기준연도' in df.columns:
                df['기준연도'] = pd.to_numeric(df['기준연도'], errors='coerce')
                # Get latest year data
                latest_year = df['기준연도'].max()
                latest = df[df['기준연도'] == latest_year].copy()
                # Extract image text and percentage
                # "연상이미지(%)" contains text like "한국음식(29.2)"
                row_data = {'country': country}
                for _, r in latest.iterrows():
                    img_text = str(r.get('연상이미지(%)', ''))
                    rank = r.get('순위', 0)
                    # Extract percentage from text like "한국음식(29.2)" or just numeric
                    pct_match = re.search(r'\((\d+\.?\d*)\)', img_text)
                    name_match = re.match(r'([^(]+)', img_text)
                    if pct_match and name_match:
                        name = name_match.group(1).strip()
                        pct = float(pct_match.group(1))
                        if rank == 1:
                            row_data['image_top1_pct'] = pct
                        # Check for K-food, K-pop categories
                        if any(k in name for k in ['음식','푸드','food','Food']):
                            row_data['image_kfood_pct'] = pct
                        if any(k in name for k in ['K-POP','케이팝','K-pop','kpop','팝']):
                            row_data['image_kpop_pct'] = pct
                        if any(k in name for k in ['뷰티','화장품','미용','Beauty']):
                            row_data['image_kbeauty_pct'] = pct
                        if any(k in name for k in ['드라마','영화','Drama','drama']):
                            row_data['image_kdrama_pct'] = pct
                frames.append(row_data)
        if not frames: return pd.DataFrame()
        result = pd.DataFrame(frames)
        # Fill missing with 0
        for col in ['image_top1_pct','image_kfood_pct','image_kpop_pct','image_kbeauty_pct','image_kdrama_pct']:
            if col not in result.columns:
                result[col] = 0
            result[col] = result[col].fillna(0)
        return result

    # ── NEW: Visit Purpose ──
    def load_visit_purpose(self):
        """국가별 방문목적 비율 (관광%, 비즈니스%)"""
        frames = []
        for country, dirpath in self.country_dirs.items():
            if country == '__global__': continue
            fpath = _find_file(dirpath, '목적별')
            if not fpath: continue
            df = _read_csv_safe(fpath)
            if df is None or df.empty: continue
            df.columns = [_nfc(c) for c in df.columns]
            df['country'] = country
            row_data = {'country': country}
            if '구분' in df.columns and '비율' in df.columns:
                df['비율'] = pd.to_numeric(df['비율'].astype(str).str.replace(',',''), errors='coerce')
                for _, r in df.iterrows():
                    purpose = _nfc(str(r['구분']))
                    pct = r['비율']
                    if '여가' in purpose or '관광' in purpose or '위락' in purpose:
                        row_data['tourism_pct'] = pct
                    elif '사업' in purpose or '비즈니스' in purpose or '상용' in purpose:
                        row_data['business_pct'] = pct
            frames.append(row_data)
        if not frames: return pd.DataFrame()
        result = pd.DataFrame(frames)
        for col in ['tourism_pct','business_pct']:
            if col not in result.columns: result[col] = np.nan
        return result

    # ── NEW: Weather ──
    def load_weather(self):
        """월별 날씨 데이터 (서울 + 전국 평균)"""
        if not self.project_path:
            return pd.DataFrame()
        weather_dir = os.path.join(self.project_path, 'data', 'weather')
        if not os.path.exists(weather_dir):
            return pd.DataFrame()
        fpath = None
        for entry in os.scandir(weather_dir):
            if 'regional' in _nfc(entry.name).lower() and entry.name.endswith('.csv'):
                fpath = entry.path
                break
        if not fpath:
            for entry in os.scandir(weather_dir):
                if entry.name.endswith('.csv'):
                    fpath = entry.path
                    break
        if not fpath:
            return pd.DataFrame()
        df = _read_csv_safe(fpath)
        if df is None or df.empty:
            return pd.DataFrame()
        # Make year_month
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].apply(lambda x: f'{x:02d}')
        # Seoul data
        seoul = df[df['region'].str.contains('Seoul|서울|Incheon|인천', case=False, na=False)].copy()
        if seoul.empty:
            seoul = df.groupby('year_month').first().reset_index()
        seoul = seoul[['year_month','avg_temp_c','precipitation_mm']].copy()
        seoul = seoul.rename(columns={'avg_temp_c':'seoul_temp','precipitation_mm':'seoul_precip'})
        # National average
        nat = df.groupby('year_month').agg(
            nat_temp=('avg_temp_c','mean'),
            nat_precip=('precipitation_mm','sum'),
        ).reset_index()
        result = seoul.merge(nat, on='year_month', how='outer')
        return result

    # ── NEW: Events ──
    def load_events(self):
        """월별 이벤트/축제 데이터 (전국 + 서울)"""
        if not self.project_path:
            return pd.DataFrame()
        holiday_dir = os.path.join(self.project_path, 'data', 'holiday')
        if not os.path.exists(holiday_dir):
            return pd.DataFrame()
        
        all_events = []
        for entry in os.scandir(holiday_dir):
            name = _nfc(entry.name)
            if 'event_all_region' in name and name.endswith('.xlsx'):
                year_match = re.search(r'(\d{4})', name)
                if not year_match: continue
                year = int(year_match.group(1))
                try:
                    # Read 세부현황 sheet for detailed data
                    df = pd.read_excel(entry.path, sheet_name='세부현황')
                    df.columns = [_nfc(str(c)).strip() for c in df.columns]
                    df['file_year'] = year
                    all_events.append(df)
                except Exception:
                    try:
                        # Read 총괄 sheet as fallback
                        df_summary = pd.read_excel(entry.path, sheet_name='총괄')
                        # 총괄 has region totals — extract Seoul and total
                        # Structure varies, just store year-level counts
                        all_events.append(pd.DataFrame({
                            'file_year': [year],
                            'event_count_total': [df_summary.shape[0]],
                        }))
                    except:
                        pass
        
        if not all_events:
            return pd.DataFrame()
        
        events = pd.concat(all_events, ignore_index=True)
        
        # Try to extract monthly event counts
        # Check for date columns - 개최기간 has start/end dates
        result_rows = []
        if '개최기간' in events.columns or '축제시작일자' in events.columns:
            date_col = '개최기간' if '개최기간' in events.columns else '축제시작일자'
            region_col = None
            for c in events.columns:
                if '광역' in c or '지역' in c or '시도' in c:
                    region_col = c
                    break
            
            for _, row in events.iterrows():
                date_str = str(row.get(date_col, ''))
                year = row.get('file_year', 2023)
                region = str(row.get(region_col, '')) if region_col else ''
                
                # Extract month from date string
                month_match = re.search(r'(\d{1,2})\.', date_str) or re.search(r'(\d{1,2})월', date_str)
                if not month_match:
                    month_match = re.search(r'\d{4}[.-](\d{1,2})', date_str)
                
                if month_match:
                    month = int(month_match.group(1))
                    if 1 <= month <= 12:
                        is_seoul = 1 if '서울' in region else 0
                        result_rows.append({
                            'year': year, 'month': month,
                            'is_seoul': is_seoul,
                        })
        
        if not result_rows:
            # Fallback: just count events per year
            yearly = events.groupby('file_year').size().reset_index(name='event_count')
            yearly = yearly.rename(columns={'file_year':'year'})
            # Distribute evenly across months
            rows = []
            for _, r in yearly.iterrows():
                for m in range(1,13):
                    rows.append({'year':int(r['year']),'month':m,
                                'event_count_national':r['event_count']/12,
                                'event_count_seoul':r['event_count']/12*0.1})
            result = pd.DataFrame(rows)
            result['year_month'] = result['year'].astype(str) + '-' + result['month'].apply(lambda x: f'{x:02d}')
            return result[['year_month','event_count_national','event_count_seoul']]
        
        evt_df = pd.DataFrame(result_rows)
        # Aggregate: monthly national count + Seoul count
        national = evt_df.groupby(['year','month']).size().reset_index(name='event_count_national')
        seoul_evts = evt_df[evt_df['is_seoul']==1].groupby(['year','month']).size().reset_index(name='event_count_seoul')
        result = national.merge(seoul_evts, on=['year','month'], how='left')
        result['event_count_seoul'] = result['event_count_seoul'].fillna(0)
        result['year_month'] = result['year'].astype(str) + '-' + result['month'].apply(lambda x: f'{x:02d}')
        return result[['year_month','event_count_national','event_count_seoul']]

    # ── Build full panel ──
    def build_panel(self):
        print("[1/8] Loading monthly visitors...")
        panel = self.load_monthly_visitors()
        panel['country_group'] = panel['country'].apply(get_country_group)
        panel['year'] = panel['year_month'].str[:4].astype(int)
        panel['month'] = panel['year_month'].str[5:7].astype(int)
        print(f"  → {len(panel)} rows, {panel['country'].nunique()} countries")

        print("[2/8] Loading annual satisfaction...")
        sat = self.load_annual_satisfaction()
        if not sat.empty and 'year' in sat.columns:
            sat['year'] = sat['year'].astype(int)
            # Filter out COVID anomalies in satisfaction (2020-2021 had extreme values)
            sat_cols = ['revisit_rate','stay_days','per_capita_spend_usd','daily_spend_usd',
                       'overall_satisfaction','revisit_intention','recommend_intention']
            merge_cols = [c for c in sat_cols if c in sat.columns]
            if merge_cols:
                panel = panel.merge(sat[['country','year']+merge_cols], on=['country','year'], how='left')
                # Cap COVID-era outliers
                for col in ['stay_days','per_capita_spend_usd','daily_spend_usd']:
                    if col in panel.columns:
                        covid_mask = panel['year_month'].between('2020-01','2022-06')
                        panel.loc[covid_mask, col] = np.nan

        print("[3/8] Loading travel interest...")
        interest = self.load_travel_interest()
        if not interest.empty:
            ci = interest[interest['country'] != '__global__']
            if not ci.empty:
                panel = panel.merge(ci[['year_month','country','travel_interest_pct']],
                                   on=['year_month','country'], how='left')
            gi = interest[interest['country']=='__global__'][['year_month','travel_interest_pct']]
            gi = gi.rename(columns={'travel_interest_pct':'global_travel_interest_pct'})
            if not gi.empty:
                panel = panel.merge(gi, on='year_month', how='left')

        print("[4/8] Loading Hallyu interest...")
        hallyu = self.load_hallyu_interest()
        if not hallyu.empty:
            panel = panel.merge(hallyu[['country','hallyu_index']], on='country', how='left')

        print("[5/8] Loading Korea image...")
        image = self.load_korea_image()
        if not image.empty:
            panel = panel.merge(image, on='country', how='left')

        print("[6/8] Loading visit purpose...")
        purpose = self.load_visit_purpose()
        if not purpose.empty:
            panel = panel.merge(purpose, on='country', how='left')

        print("[7/8] Loading weather...")
        weather = self.load_weather()
        if not weather.empty:
            panel = panel.merge(weather, on='year_month', how='left')

        print("[8/8] Loading events...")
        events = self.load_events()
        if not events.empty:
            panel = panel.merge(events, on='year_month', how='left')

        panel = panel.sort_values(['country','year_month']).reset_index(drop=True)
        print(f"\n✅ V4 Panel: {panel.shape[0]} rows × {panel.shape[1]} columns")
        print(f"   Countries: {panel['country'].nunique()}")
        print(f"   Date range: {panel['year_month'].min()} ~ {panel['year_month'].max()}")
        return panel
