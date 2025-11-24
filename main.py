import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import pandas as pd
import numpy as np
import json
import os
import webbrowser
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
# Optional Dependency Check
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# ==============================================================================
#  GLOBAL THEME CONFIGURATION (Forensic Dark Mode)
# ==============================================================================
THEME = {
    "bg": "#050505",          # Pure Void
    "panel": "#121212",       # Obsidian
    "fg": "#e0e0e0",          # Platinum
    "accent": "#00f0ff",      # Cyber Cyan (Utility)
    "shock": "#ff003c",       # Critical Red (Friction/Risk)
    "warn": "#ffea00",        # Amber (Warning)
    "safe": "#00ff9d",        # Spring Green (Growth)
    "l2": "#9d00ff",          # Purple (Lightning/L2)
    "grid": "#1f1f1f",        # Subtle Grid
    "font_h": ("Segoe UI", 11, "bold"),
    "font_t": ("Consolas", 9)
}

plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": THEME["bg"],
    "figure.facecolor": THEME["bg"],
    "grid.color": THEME["grid"],
    "text.color": THEME["fg"],
    "xtick.color": "#666",
    "ytick.color": "#666",
    "axes.edgecolor": "#333",
    "savefig.facecolor": THEME["bg"],
    "savefig.edgecolor": THEME["bg"]
})

# ==============================================================================
#  1. BULLETPROOF DATA PARSER & GENERATOR
# ==============================================================================
class DataHandler:
    """
    Combines the robust parsing of Code 2 with the rich dummy data generation of Code 1
    to support all visualization types, now including UTXO Logic.
    """
    
    @staticmethod
    def generate_dummy_data():
        """
        Generates synthetic data that models the 'Endogenous Destabilizer' 
        with L2/Realized Cap logic, Mempool physics, and UTXO Set Dynamics.
        """
        dates = pd.date_range(start="2016-01-01", end="2025-11-22", freq='D')
        n = len(dates)
        
        # 1. Market Data & Realized Cap Logic
        returns = np.random.normal(0.0005, 0.03, n)
        price = 400 * np.cumprod(1 + returns)
        mvrv = np.sin(np.linspace(0, 20, n)) + 2.0 + np.random.normal(0, 0.1, n)
        
        # 2. Fees & Insolvency
        base_fee = np.random.lognormal(1, 0.8, n)
        min_fee = np.ones(n)
        
        # 3. UTXO Base Trend (Linear Growth 40M -> 150M)
        utxo_trend = np.linspace(40000000, 150000000, n)
        utxo_noise = np.random.normal(0, 500000, n)
        utxo_count = utxo_trend + utxo_noise
        
        # Inject Shocks
        is_shock = np.zeros(n)
        for _ in range(12):
            start = np.random.randint(50, n-100)
            duration = np.random.randint(14, 60)
            
            # Fee Spike
            base_fee[start:start+duration] *= np.linspace(2, 25, duration)
            min_fee[start:start+duration] = np.random.uniform(5, 50, duration)
            is_shock[start:start+duration] = 1
            
            # UTXO Logic: Pre-2023 (Consolidation), Post-2023 (Bloat)
            curr_date = dates[start]
            if curr_date.year < 2023:
                # Retail Exodus/Consolidation -> UTXO Count Drops
                drop_curve = np.linspace(0, -2000000, duration)
                utxo_count[start:start+duration] += drop_curve
                # Persist the drop (Hysteresis)
                utxo_count[start+duration:] -= 2000000
            else:
                # Ordinal Bloat -> UTXO Count Explodes
                bloat_curve = np.linspace(0, 5000000, duration)
                utxo_count[start:start+duration] += bloat_curve
                # Persist the bloat
                utxo_count[start+duration:] += 5000000
            
        fees = base_fee
        
        # 4. Mempool & Delay (Physical Constraints for Event Horizon Plot)
        mempool_size = fees * np.random.uniform(50000, 200000, n)
        delay = (mempool_size / 100000) + np.random.normal(10, 2, n)
        delay = np.maximum(delay, 10)
        
        # 5. L2 & Volume Logic
        ln_cap = np.linspace(0, 5000, n) * (1 + np.random.normal(0, 0.05, n))
        organic_trend = np.linspace(1000, 50000, n)
        friction_drag = pd.Series(fees).rolling(30).mean() * 50
        organic = organic_trend - friction_drag + np.random.normal(0, 500, n)
        organic = np.maximum(organic, 100)
        
        volume = (price * organic * 0.5) + (fees * price * 100)
        nvt = (price * 19000000) / volume

        df = pd.DataFrame({
            'Date': dates,
            'Price': price,
            'Fees': fees,
            'Delay': delay,
            'Volume': volume,
            'Mempool': mempool_size,
            'Organic': organic,
            'NVT': nvt,
            'LN_Cap': ln_cap,
            'Min_Fee': min_fee,
            'MVRV': mvrv,
            'UTXO': utxo_count
        })
        return df.set_index('Date')

    @staticmethod
    def parse_file(filepath, hint):
        try:
            ext = os.path.splitext(filepath)[1].lower()
            df = None

            # JSON Handling
            if ext == '.json':
                with open(filepath, 'r') as f:
                    raw = json.load(f)
                target = raw
                if isinstance(raw, dict):
                    for k in raw.keys():
                        if isinstance(raw[k], list):
                            target = raw[k]; break
                data = []
                for item in target:
                    ts = item.get('x') or item.get('t') or item.get('timestamp')
                    val = item.get('y') or item.get('v') or item.get('value')
                    if isinstance(item, list): ts, val = item[0], item[1]
                    if ts is not None and val is not None:
                        try:
                            ts = float(ts)
                            if ts > 1e11: ts /= 1000
                            dt = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
                            data.append({'Date': dt, 'Value': float(val)})
                        except: pass
                df = pd.DataFrame(data)

            # CSV Handling
            elif ext == '.csv':
                df = pd.read_csv(filepath, sep=None, engine='python')
                df.columns = [str(c).strip().lower() for c in df.columns]
                date_col = next((c for c in df.columns if any(x in c for x in ['date', 'time', 'day', 'ts'])), None)
                if not date_col: raise ValueError("No Date column found")
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                
                cols = [c for c in df.columns if c != date_col]
                target_col = None
                hint_map = {
                    'ln_cap': ['btc', 'cap', 'amount'], 'fees': ['usd', 'cost', 'fee'],
                    'price': ['usd', 'val', 'price'], 'min_fee': ['sat', 'vbyte', 'fee'],
                    'utxo': ['count', 'output', 'utxo']
                }
                search_terms = hint_map.get(hint.lower(), ['val', 'usd', 'btc'])
                for term in search_terms:
                    target_col = next((c for c in cols if term in c), None)
                    if target_col: break
                if not target_col: target_col = cols[-1]
                
                if df[target_col].dtype == 'object':
                    df[target_col] = df[target_col].astype(str).str.replace(',', '', regex=True)
                df['Value'] = pd.to_numeric(df[target_col], errors='coerce')

            if df is not None and not df.empty:
                df = df.dropna(subset=['Date', 'Value'])
                df = df.set_index('Date')
                df = df[['Value']].resample('D').mean()
                return df.rename(columns={'Value': hint})

        except Exception as e:
            print(f"Parser Error [{hint}]: {e}")
            return None
        return None

# ==============================================================================
#  2. FORENSIC ENGINE (Advanced Logic + Visualization Support)
# ==============================================================================
class ForensicEngine:
    def __init__(self, df, threshold_percentile=95.0):
        self.df = df.copy().interpolate(method='time').dropna()
        self.threshold_pct = threshold_percentile
        self._run_pipeline()

    def _run_pipeline(self):
        self._apply_logical_fixes()
        self._calc_tci()
        self._detect_regimes()
        self._calc_impact_metrics()
        self._calc_divergence() # From Code 1
        self._calc_utxo_health() # NEW: UTXO Logic
        self._calc_phase_state()

    def _apply_logical_fixes(self):
        # 1. Realized Cap (Denominator Fix)
        supply_est = 19700000
        if 'Price' in self.df.columns and 'MVRV' in self.df.columns:
            mcap = self.df['Price'] * supply_est
            self.df['Realized_Cap'] = mcap / self.df['MVRV'].replace(0, 1)
            self.cap_type = "Realized Cap (Robust)"
        elif 'Price' in self.df.columns:
            self.df['Realized_Cap'] = self.df['Price'] * supply_est
            self.cap_type = "Market Cap (Standard)"
        else:
            self.df['Realized_Cap'] = 1e9
            self.cap_type = "Synthetic"

        # 2. Velocity (L2 Fix)
        if 'Volume' in self.df.columns:
            self.df['Vol_L1'] = self.df['Volume']
            if 'LN_Cap' in self.df.columns and 'Price' in self.df.columns:
                self.df['Vol_L2'] = self.df['LN_Cap'] * self.df['Price'] * 10.0
                self.df['Vol_Total'] = self.df['Vol_L1'] + self.df['Vol_L2']
                self.vol_type = "L1 + L2 (Adjusted)"
            else:
                self.df['Vol_Total'] = self.df['Vol_L1']
                self.vol_type = "L1 Only"
            self.df['Vel_Robust'] = self.df['Vol_Total'] / self.df['Realized_Cap']
        else:
            self.df['Vel_Robust'] = np.zeros(len(self.df))
            self.vol_type = "N/A"

        if 'NVT' in self.df.columns:
            self.df['Vel_Standard'] = 1 / self.df['NVT'].replace(0, 1)
        else:
            self.df['Vel_Standard'] = self.df['Vel_Robust']

        self.df['Velocity'] = self.df['Vel_Robust'] if self.df['Vel_Robust'].sum() > 0 else self.df['Vel_Standard']
        self.df['Price_Mom'] = self.df['Price'].pct_change(30).fillna(0) if 'Price' in self.df.columns else 0

    def _calc_tci(self):
            # Advanced TCI: (Magnitude + Volatility) * Delay
            if 'Fees' in self.df.columns:
                capped_fees = self.df['Fees'].clip(upper=self.df['Fees'].quantile(0.99))
                fee_mean = capped_fees.rolling(30).mean().bfill()
                fee_std = capped_fees.rolling(30).std().fillna(0)
                fee_stress = fee_mean + fee_std
                
                if 'Min_Fee' in self.df.columns:
                    self.df['Insolvency'] = self.df['Min_Fee'] 
                    
                    # If median > 50, assume Cents or Sats
                    if self.df['Min_Fee'].median() > 50:
                         self.insol_lbl = "Cost (US Cents)" 
                         self.insol_thresh = 1000.0       # $10.00 = 1000 cents
                    else:
                         self.insol_lbl = "Min Fee (sat/vB)"
                         self.insol_thresh = 5.0
                else:
                    self.df['Insolvency'] = self.df['Fees']
                    self.insol_lbl = "Avg Fee (USD)"
                    self.insol_thresh = 10.0
            else:
                fee_stress = 0
                self.df['Insolvency'] = 0
                self.insol_lbl = "N/A"
                self.insol_thresh = 0

            # Alias for Plotting compatibility
            self.df['RCI'] = self.df['Insolvency']

            delay_factor = (self.df['Delay'] / 10.0) if 'Delay' in self.df.columns else 1.0
            self.df['TCI'] = fee_stress * delay_factor
            self.df['TCI'] = self.df['TCI'].fillna(0)
            
    def _detect_regimes(self):
        self.thresh_val = self.df['TCI'].quantile(self.threshold_pct / 100.0)
        self.df['Regime'] = np.where(self.df['TCI'] >= self.thresh_val, 'SHOCK', 'NORMAL')

    def _calc_impact_metrics(self):
        self.df['Vel_Next_30d'] = self.df['Velocity'].shift(-30)
        self.df['Vel_30d_Change'] = ((self.df['Vel_Next_30d'] - self.df['Velocity']) / self.df['Velocity']) * 100.0
        self.df['Vel_Mom'] = self.df['Velocity'].pct_change(30)
        
        conditions = [
            (self.df['Price_Mom'] > 0) & (self.df['Vel_Mom'] > 0),
            (self.df['Price_Mom'] > 0) & (self.df['Vel_Mom'] <= 0),
            (self.df['Price_Mom'] <= 0) & (self.df['Vel_Mom'] <= 0),
            (self.df['Price_Mom'] <= 0) & (self.df['Vel_Mom'] > 0)
        ]
        self.df['State'] = np.select(conditions, ['Boom', 'Speculation', 'Stagflation', 'Capitulation'], default='Neutral')

    def _calc_divergence(self):
        # From Code 1: Z-Score Divergence
        if 'Organic' in self.df.columns and 'Volume' in self.df.columns:
            scaler = StandardScaler()
            vol_z = scaler.fit_transform(self.df[['Volume']].fillna(0))
            org_z = scaler.fit_transform(self.df[['Organic']].fillna(0))
            self.df['Div_Score'] = vol_z - org_z

    def _calc_utxo_health(self):
        # Calculate the rate of change in the UTXO set
        if 'UTXO' in self.df.columns:
            self.df['UTXO_Growth'] = self.df['UTXO'].pct_change(30)
            
            # THE INSOLVENCY CHECK:
            # If Fees are High (Shock) AND UTXO Growth is Negative -> Confirmed Retail Exodus
            condition_exodus = (self.df['Regime'] == 'SHOCK') & (self.df['UTXO_Growth'] < 0)
            self.df['Confirmed_Exodus'] = np.where(condition_exodus, 1, 0)
            
            # THE ORDINAL NOISE (2023+):
            # If Fees are High AND UTXO Growth is Massive -> Spam/Bloat (Not Adoption)
            condition_bloat = (self.df['Regime'] == 'SHOCK') & (self.df['UTXO_Growth'] > 0.05)
            self.df['State_Bloat'] = np.where(condition_bloat, 1, 0)
        else:
            self.df['UTXO_Growth'] = 0
            self.df['Confirmed_Exodus'] = 0
            self.df['State_Bloat'] = 0

    def _calc_phase_state(self):
        self.df['TCI_Log'] = np.log1p(self.df['TCI'])
        self.df['Vel_Log'] = np.log1p(self.df['Velocity'])

# ==============================================================================
#  3. ECONOMETRIC VALIDATOR
# ==============================================================================
class EconometricValidator:
    """
    Implements Appendix A (Mathematical Derivations) and Appendix B (Statistical Tests)
    Strict implementation for Forensic Report generation.
    """
    
    @staticmethod
    def run_log_log_elasticity(df):
        # Appendix A.4: Log-Log Elasticity Model
        try:
            # Shift Velocity back 30 days to align with TCI
            dataset = pd.DataFrame({
                'ln_V': np.log(df['Velocity'].shift(-30)), 
                'ln_TCI': np.log(df['TCI'].replace(0, 0.0001)) 
            }).dropna().replace([np.inf, -np.inf], 0)
            
            if len(dataset) < 10: raise ValueError
            
            X = sm.add_constant(dataset['ln_TCI'])
            y = dataset['ln_V']
            model = sm.OLS(y, X).fit()
            return {
                'beta': model.params['ln_TCI'], 
                'r_squared': model.rsquared,
                'p_value': model.pvalues['ln_TCI']
            }
        except:
            return {'beta': 0.0, 'p_value': 1.0, 'r_squared': 0.0}

    @staticmethod
    def run_threshold_regression(df):
        # Appendix A.3: Threshold Regression Model
        try:
            dataset = df.copy().dropna(subset=['Vel_30d_Change'])
            dataset['S_t'] = np.where(dataset['Regime'] == 'SHOCK', 1, 0)
            
            if len(dataset) < 10: raise ValueError
            
            X = sm.add_constant(dataset['S_t'])
            y = dataset['Vel_30d_Change']
            model = sm.OLS(y, X).fit()
            
            return {
                'beta_1_normal': model.params['const'], 
                'marginal_impact': model.params['S_t'],
                'beta_2_shock': model.params['const'] + model.params['S_t'],
                'p_value': model.pvalues['S_t']
            }
        except:
            return {'beta_1_normal': 0.0, 'marginal_impact': 0.0, 'beta_2_shock': 0.0, 'p_value': 1.0}

    @staticmethod
    def run_welchs_t_test(df):
        # Appendix B.1: Welch's t-Test
        try:
            normal = df[df['Regime'] == 'NORMAL']['Vel_30d_Change'].dropna()
            shock = df[df['Regime'] == 'SHOCK']['Vel_30d_Change'].dropna()
            if len(normal) < 2 or len(shock) < 2: return 0.0, 1.0
            t_stat, p_val = stats.ttest_ind(normal, shock, equal_var=False)
            return t_stat, p_val
        except:
            return 0.0, 1.0

    @staticmethod
    def run_bootstrap_ci(df, iterations=5000):
        # Appendix B.2: Bootstrap Analysis
        try:
            normal = df[df['Regime'] == 'NORMAL']['Vel_30d_Change'].dropna().values
            shock = df[df['Regime'] == 'SHOCK']['Vel_30d_Change'].dropna().values
            
            if len(shock) < 5: return 0.0, 0.0
            
            n_norm, n_shock = len(normal), len(shock)
            idx_n = np.random.randint(0, n_norm, (iterations, n_norm))
            idx_s = np.random.randint(0, n_shock, (iterations, n_shock))
            
            diffs = np.mean(shock[idx_s], axis=1) - np.mean(normal[idx_n], axis=1)
            return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        except:
            return 0.0, 0.0
# ==============================================================================
#  4. REPORT GENERATOR
# ==============================================================================
class ReportGenerator:
    @staticmethod
    def generate(engine, events_df, regime_stats, insol_stats, validation, granger_html):
        df = engine.df
        
        # Helper for p-value formatting
        def fmt_p(p): return "< 0.0001" if p < 0.0001 else f"{p:.4f}"
        
        # Unpack Validation Results
        elast = validation['elasticity']
        thresh = validation['threshold']
        welch = validation['welch']
        boot = validation['bootstrap']
        
        sig_verdict = "SIGNIFICANT (p < 0.05)" if welch[1] < 0.05 else "INCONCLUSIVE (Tail Noise)"

        # UTXO Stats (Basic)
        if 'UTXO' in df.columns:
            utxo_normal_growth = df[df['Regime']=='NORMAL']['UTXO_Growth'].mean() * 100
            utxo_shock_growth = df[df['Regime']=='SHOCK']['UTXO_Growth'].mean() * 100
        else:
            utxo_normal_growth = 0
            utxo_shock_growth = 0

        # ==============================================================================
        #  FORENSIC DEEP DIVE: THE UTXO QUALITY TEST CALCULATIONS
        # ==============================================================================
        utxo_html = ""
        if 'UTXO' in df.columns and 'Realized_Cap' in df.columns:
            # 1. Regime Correlation Split (Pre/Post 2023)
            # Ensure index is sorted for slicing
            df_sorted = df.sort_index()
            pre_2023 = df_sorted[df_sorted.index < '2023-01-01']
            post_2023 = df_sorted[df_sorted.index >= '2023-01-01']
            
            # Correlation between Friction (TCI) and UTXO Growth
            # We look for sign inversion: Negative (Exodus) -> Positive (Spam)
            corr_pre = pre_2023['TCI'].corr(pre_2023['UTXO_Growth'])
            corr_post = post_2023['TCI'].corr(post_2023['UTXO_Growth'])
            
            # 2. UTXO Value Dilution (Realized Cap / Count)
            # Compare Avg UTXO Value during Normal vs Shock
            # Formula: (Realized Cap / UTXO Count)
            df['Val_Per_UTXO'] = df['Realized_Cap'] / df['UTXO'].replace(0, 1)
            avg_val_normal = df[df['Regime']=='NORMAL']['Val_Per_UTXO'].mean()
            avg_val_shock = df[df['Regime']=='SHOCK']['Val_Per_UTXO'].mean()
            
            dilution_pct = ((avg_val_shock - avg_val_normal) / avg_val_normal) * 100
            
            # 3. Zombie Ratio (Volume / UTXO)
            # Formula: (L1 Volume / UTXO Count) - Does activity keep up with set size?
            df['Act_Per_UTXO'] = df['Volume'] / df['UTXO'].replace(0, 1)
            act_normal = df[df['Regime']=='NORMAL']['Act_Per_UTXO'].mean()
            act_shock = df[df['Regime']=='SHOCK']['Act_Per_UTXO'].mean()
            
            zombie_impact = ((act_shock - act_normal) / act_normal) * 100

            # Determine Verdicts
            verdict_corr = '<strong>STRUCTURAL INVERSION</strong><br>(Exodus &rarr; Spam)' if (corr_pre < 0 and corr_post > 0) else 'Inconclusive/Linear'
            verdict_dilution = '<strong>DUST CONFIRMED</strong><br>(High Count / Low Value)' if dilution_pct < -5 else 'Healthy Growth'
            verdict_zombie = '<strong>ACTIVITY COLLAPSE</strong><br>(Ledger Bloat)' if zombie_impact < -10 else 'Active Userbase'

            # Build the Deep Dive HTML Block
            utxo_html = f"""
            <h2>6. Deep Dive: UTXO Quality Assurance</h2>
            <p>Mathematical isolation of "Adoption" vs. "Bloat" using Regime-Split Correlation and Unit Economics.</p>
            <table>
                <thead>
                    <tr><th>Forensic Test</th><th>Result</th><th>Verdict</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <strong>Regime Correlation Split</strong><br>
                            <span style="color:#888; font-size:10px;">Correlation: Friction (TCI) vs. UTXO Growth</span>
                        </td>
                        <td class="mono">
                            Pre-2023: <span class="{'success' if corr_pre < 0 else 'danger'}">{corr_pre:.4f}</span><br>
                            Post-2023: <span class="{'danger' if corr_post > 0 else 'success'}">{corr_post:.4f}</span>
                        </td>
                        <td class="{'danger' if (corr_pre < 0 and corr_post > 0) else 'mono'}">{verdict_corr}</td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Value Dilution</strong><br>
                            <span style="color:#888; font-size:10px;">Change in Realized Value per UTXO during Shock</span>
                        </td>
                        <td class="mono { 'danger' if dilution_pct < 0 else 'success'}">{dilution_pct:+.2f}%</td>
                        <td class="{ 'danger' if dilution_pct < -5 else 'success'}">{verdict_dilution}</td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Zombie Ratio</strong><br>
                            <span style="color:#888; font-size:10px;">Change in Volume per UTXO during Shock</span>
                        </td>
                        <td class="mono { 'danger' if zombie_impact < 0 else 'success'}">{zombie_impact:+.2f}%</td>
                        <td class="{ 'danger' if zombie_impact < -10 else 'success'}">{verdict_zombie}</td>
                    </tr>
                </tbody>
            </table>
            """
        elif 'UTXO' not in df.columns:
            utxo_html = "<h2>6. Deep Dive: UTXO Quality Assurance</h2><p style='color:#666;'><em>UTXO Data not loaded. Load 'UTXO' file to enable structural quality tests.</em></p>"
        else:
             utxo_html = "<h2>6. Deep Dive: UTXO Quality Assurance</h2><p style='color:#666;'><em>Realized Cap data insufficient for unit economic analysis.</em></p>"


        # Construct Main HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Audit: Endogenous Destabilizer</title>
            <style>
                :root {{ --bg: #050505; --panel: #121212; --text: #e0e0e0; --accent: #00f0ff; --danger: #ff003c; --success: #00ff9d; --warn: #ffea00; }}
                body {{ background-color: var(--bg); color: var(--text); font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 40px; font-size: 12px; max-width: 1400px; margin: 0 auto; }}
                h1 {{ color: var(--accent); border-bottom: 2px solid var(--danger); padding-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; font-size: 24px; }}
                h2 {{ color: #fff; background: linear-gradient(90deg, #222, #000); padding: 10px; border-left: 4px solid var(--accent); margin-top: 40px; text-transform: uppercase; font-size: 16px; }}
                h3 {{ color: #888; margin-top: 5px; font-weight: normal; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                .kpi-card {{ background: var(--panel); padding: 15px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }}
                .kpi-val {{ font-size: 22px; font-weight: 700; color: #fff; display: block; margin-top: 5px; }}
                .kpi-lbl {{ color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 11px; background: var(--panel); }}
                th {{ background: #000; color: var(--accent); padding: 12px 8px; text-align: left; border-bottom: 2px solid #333; text-transform: uppercase; font-weight: 600; }}
                td {{ padding: 8px; border-bottom: 1px solid #2a2a2a; color: #ccc; }}
                tr:nth-child(even) {{ background-color: #181818; }}
                .danger {{ color: var(--danger); font-weight: bold; }}
                .success {{ color: var(--success); font-weight: bold; }}
                .warn {{ color: var(--warn); font-weight: bold; }}
                .mono {{ font-family: 'Consolas', monospace; }}
            </style>
        </head>
        <body>
            <h1>Forensic Blockchain Audit: Combined Logic v7.0</h1>
            <h3>Analysis Engine | Cap Logic: {engine.cap_type} | Vol Logic: {engine.vol_type}</h3>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Threshold: Top {100-engine.threshold_pct:.1f}%</p>

            <!-- EXECUTIVE DASHBOARD -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-lbl">Critical Events</span>
                    <span class="kpi-val" style="color:{'#ff003c' if len(events_df) > 0 else '#00ff9d'}">
                        {len(events_df)}
                    </span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Net Velocity Damage</span>
                    <span class="kpi-val danger">
                        {regime_stats.loc['SHOCK', 'Vel_30d_Change'] - regime_stats.loc['NORMAL', 'Vel_30d_Change']:+.2f}%
                    </span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Stagflation Risk</span>
                    <span class="kpi-val danger">{insol_stats['stag_prob']:.1f}%</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Welch's P-Value</span>
                    <span class="kpi-val mono" style="font-size:14px; padding-top:8px;">{fmt_p(welch[1])}</span>
                </div>
            </div>

            <!-- SECTION 1: ECONOMETRIC VALIDATION (APPENDIX A & B) -->
            <h2>1. Econometric Validation (Appendix A & B)</h2>
            <p>Mathematical verification of the Endogenous Destabilizer Hypothesis using OLS and Bootstrap methodology.</p>
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <div>
                    <h3>Appendix A.3: Threshold Regression</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Beta 1 (Normal Growth)</td><td class="success mono">{thresh['beta_1_normal']:.4f}%</td></tr>
                        <tr><td>Beta 2 (Shock Growth)</td><td class="danger mono">{thresh['beta_2_shock']:.4f}%</td></tr>
                        <tr><td><strong>Marginal Impact</strong></td><td class="danger mono"><strong>{thresh['marginal_impact']:.4f}%</strong></td></tr>
                    </table>
                </div>
                <div>
                    <h3>Appendix A.4: Log-Log Elasticity</h3>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Elasticity (Beta)</td><td class="mono">{elast['beta']:.4f}</td></tr>
                        <tr><td>R-Squared</td><td class="mono">{elast['r_squared']:.4f}</td></tr>
                        <tr><td>P-Value</td><td class="mono">{fmt_p(elast['p_value'])}</td></tr>
                    </table>
                </div>
            </div>

            <h3>Appendix B: Statistical Significance Tests</h3>
            <table>
                <thead><tr><th>Test Method</th><th>Result</th><th>Verdict</th></tr></thead>
                <tbody>
                    <tr>
                        <td>Welch's t-Test (Unequal Variance)</td>
                        <td class="mono">t-stat: {welch[0]:.4f} | p-val: {fmt_p(welch[1])}</td>
                        <td class="{'success' if welch[1] < 0.05 else 'warn'}">{sig_verdict}</td>
                    </tr>
                    <tr>
                        <td>Bootstrap 95% CI (5,000 Iterations)</td>
                        <td class="mono">[{boot[0]:.2f}%, {boot[1]:.2f}%]</td>
                        <td>{'VALID (Non-Zero)' if (boot[0] < 0 and boot[1] < 0) else 'INCONCLUSIVE'}</td>
                    </tr>
                </tbody>
            </table>

            <!-- SECTION 2: STRUCTURAL REGIME ANALYSIS -->
            <h2>2. Structural Regime Analysis</h2>
            <p>Comparative analysis of network physics between 'Normal' operations and 'Shock' friction regimes.</p>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Normal State (Avg)</th>
                        <th>Shock State (Avg)</th>
                        <th>Multiplier / Delta</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Avg Transaction Fee</td>
                        <td>${regime_stats.loc['NORMAL', 'Fees']:.2f}</td>
                        <td class="danger">${regime_stats.loc['SHOCK', 'Fees']:.2f}</td>
                        <td class="warn">{regime_stats.loc['SHOCK', 'Fees'] / regime_stats.loc['NORMAL', 'Fees']:.1f}x Increase</td>
                    </tr>
                    <tr>
                        <td>Avg Delay (Latency)</td>
                        <td>{regime_stats.loc['NORMAL', 'Delay']:.1f} min</td>
                        <td class="danger">{regime_stats.loc['SHOCK', 'Delay']:.1f} min</td>
                        <td class="warn">{regime_stats.loc['SHOCK', 'Delay'] / regime_stats.loc['NORMAL', 'Delay']:.1f}x Slower</td>
                    </tr>
                    <tr>
                        <td>Transaction Cost Index (TCI)</td>
                        <td>{regime_stats.loc['NORMAL', 'TCI']:.2f}</td>
                        <td class="danger">{regime_stats.loc['SHOCK', 'TCI']:.2f}</td>
                        <td class="danger">{regime_stats.loc['SHOCK', 'TCI'] / regime_stats.loc['NORMAL', 'TCI']:.1f}x Stress</td>
                    </tr>
                    <tr>
                        <td>UTXO Set Growth (30d)</td>
                        <td class="success">{utxo_normal_growth:+.2f}%</td>
                        <td class="warn">{utxo_shock_growth:+.2f}%</td>
                        <td>Structural Change</td>
                    </tr>
                    <tr>
                        <td>30-Day Velocity Growth</td>
                        <td class="success">{regime_stats.loc['NORMAL', 'Vel_30d_Change']:+.2f}%</td>
                        <td class="danger">{regime_stats.loc['SHOCK', 'Vel_30d_Change']:+.2f}%</td>
                        <td class="danger">CRITICAL FAILURE</td>
                    </tr>
                </tbody>
            </table>

            <!-- SECTION 3: INSOLVENCY ANALYSIS -->
            <h2>3. Insolvency & Causality</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Value</th><th>Implication</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Insolvency Days</td>
                        <td class="mono">{insol_stats['days']} Days</td>
                        <td>Direct Retail Lockout (>{engine.insol_lbl})</td>
                    </tr>
                    <tr>
                        <td>Peak Cost Metric</td>
                        <td class="warn mono">{insol_stats['max']:.2f}</td>
                        <td>Exceeded {engine.insol_thresh} Threshold</td>
                    </tr>
                    {granger_html}
                </tbody>
            </table>

            <!-- SECTION 4: DEEP DIVE CALCULATIONS (NEW) -->
            {utxo_html}

            <!-- SECTION 5: EVENT LOG -->
            <h2>4. Forensic Event Log (Aggregated)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Start Date</th><th>End Date</th><th>Duration</th><th>Peak TCI</th><th>Avg Fee</th><th>Price Imp</th><th>Vel Imp</th><th>State</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr>
                        <td class="mono">{r['Start'].strftime('%Y-%m-%d')}</td>
                        <td class="mono">{r['End'].strftime('%Y-%m-%d')}</td>
                        <td class="mono">{r['Duration']} Days</td>
                        <td class="danger mono">{r['Peak_TCI']:.1f}</td>
                        <td class="mono">${r['Avg_Fee']:.2f}</td>
                        <td class="mono {'success' if r['Price_Imp']>0 else 'danger'}">{r['Price_Imp']:+.2f}%</td>
                        <td class="mono {'success' if r['Vel_Imp']>0 else 'danger'}"><strong>{r['Vel_Imp']:+.2f}%</strong></td>
                        <td>{r['State']}</td>
                    </tr>''' for _, r in events_df.iterrows()]) if not events_df.empty else "<tr><td colspan='8'>No events detected.</td></tr>"}
                </tbody>
            </table>

            <!-- SECTION 6: DAILY LOG -->
            <h2>5. Forensic Daily Log (Raw Data)</h2>
            <div style="max-height:500px; overflow-y:scroll; border:1px solid #333;">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th><th>Fee ($)</th><th>Delay</th><th>TCI</th><th>Velocity</th><th>30d Impact</th><th>State</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td class="mono">{i.strftime('%Y-%m-%d')}</td>
                            <td class="mono">${r['Fees']:.2f}</td>
                            <td class="mono">{r['Delay']:.1f}m</td>
                            <td class="mono danger">{r['TCI']:.1f}</td>
                            <td class="mono">{r['Velocity']:.6f}</td>
                            <td class="mono {'success' if r['Vel_30d_Change']>0 else 'danger'}">{r['Vel_30d_Change']:+.2f}%</td>
                            <td>{r['State']}</td>
                        </tr>''' for i, r in df[df['Regime'] == 'SHOCK'].sort_values('Date', ascending=False).iterrows()])}
                    </tbody>
                </table>
            </div>

            <div style="margin-top:50px; border-top:1px solid #333; padding-top:10px; color:#666; text-align:center;">
                Endogenous Destabilizer Research Suite v7.0 | Generated by ForensicUnit
            </div>
        </body>
        </html>
        """
        return html
# ==============================================================================
#  5. GUI SUITE (Merged & Enhanced)
# ==============================================================================
class ResearchSuite:
    def __init__(self, root):
        self.root = root
        self.root.title("Forensic Audit Suite v7.0 (Integrated Visuals)")
        self.root.state("zoomed")
        self.root.configure(bg=THEME["bg"])
        self.file_map = {}
        self.engine = None
        self._init_ui()

    def _init_ui(self):
        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        sidebar = tk.Frame(main, bg=THEME["panel"], width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        tk.Label(sidebar, text="DATA INGESTION", font=THEME["font_h"], bg=THEME["panel"], fg=THEME["accent"]).pack(pady=(20,10), padx=15, anchor="w")
        
        inputs = ["Price", "Fees", "Delay", "Volume", "NVT", "MVRV", "LN_Cap", "Min_Fee", "Mempool", "Organic", "UTXO"]
        self.status = {}
        
        for k in inputs:
            row = tk.Frame(sidebar, bg=THEME["panel"])
            row.pack(fill=tk.X, padx=15, pady=4)
            lbl = k + (" *" if k in ['MVRV', 'LN_Cap', 'Min_Fee', 'UTXO'] else "")
            tk.Label(row, text=lbl, width=12, anchor='w', bg=THEME["panel"], fg="#ccc").pack(side=tk.LEFT)
            self.status[k] = tk.Label(row, text="-", fg="#555", bg=THEME["panel"])
            self.status[k].pack(side=tk.LEFT)
            tk.Button(row, text="LOAD", command=lambda key=k: self._load(key), bg="#222", fg="white", bd=0, width=6).pack(side=tk.RIGHT)
        
        tk.Label(sidebar, text="PARAMETERS", font=THEME["font_h"], bg=THEME["panel"], fg=THEME["accent"]).pack(pady=(30,10), padx=15, anchor="w")
        tk.Label(sidebar, text="Friction Threshold %", bg=THEME["panel"], fg="#888").pack(padx=15, anchor="w")
        self.thresh_scale = tk.Scale(sidebar, from_=80, to=99.9, orient=tk.HORIZONTAL, bg=THEME["panel"], fg=THEME["accent"], troughcolor="#222")
        self.thresh_scale.set(95.0)
        self.thresh_scale.pack(fill=tk.X, padx=15)
        
        self.btn_run = tk.Button(sidebar, text="RUN ANALYSIS", command=self._run, bg=THEME["accent"], fg="black", font=("Segoe UI", 10, "bold"), pady=8)
        self.btn_run.pack(fill=tk.X, padx=15, pady=(20, 10))
        tk.Button(sidebar, text="GENERATE DUMMY DATA", command=self._dummy, bg="#333", fg=THEME["safe"]).pack(fill=tk.X, padx=15, pady=5)
        tk.Button(sidebar, text="EXPORT REPORT", command=self._export, bg=THEME["shock"], fg="white").pack(fill=tk.X, padx=15, pady=5)
        
        self.log = scrolledtext.ScrolledText(sidebar, height=10, bg="black", fg="#0f0", font=("Consolas", 8))
        self.log.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=THEME["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["panel"], foreground="#888", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", THEME["accent"])], foreground=[("selected", "black")])
        
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _log(self, msg):
        self.log.insert(tk.END, f"> {msg}\n")
        self.log.see(tk.END)

    def _load(self, key):
        path = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.json")])
        if path:
            df = DataHandler.parse_file(path, key)
            if df is not None:
                self.file_map[key] = df
                self.status[key].config(text="OK", fg=THEME["safe"])
                self._log(f"Loaded {key}: {len(df)} rows")
            else:
                self._log(f"Failed to load {key}")

    def _dummy(self):
        df = DataHandler.generate_dummy_data()
        self.engine = ForensicEngine(df, self.thresh_scale.get())
        self._log("Synthetic Advanced Data Loaded")
        self._plot_visualizations()

    def _run(self):
        if not self.file_map: return
        self._log("Merging datasets...")
        try:
            master = list(self.file_map.values())[0]
            for d in list(self.file_map.values())[1:]:
                master = master.join(d, how='outer')
            self.engine = ForensicEngine(master, self.thresh_scale.get())
            self._log(f"Analysis Complete. Type: {self.engine.cap_type}")
            self._plot_visualizations()
        except Exception as e:
            self._log(f"Error: {e}")

    # ==========================================================================
    #  VISUALIZATION SUITE (From Code 1, Adapted for Code 2 Logic)
    # ==========================================================================
    def _plot_visualizations(self):
            self._log("--- STARTING VISUALIZATION SEQUENCE ---")
            
            # 1. DEBUG: Check Data Integrity
            df = self.engine.df
            thresh = self.engine.thresh_val
            
            self._log(f"Final Dataset Shape: {df.shape}")
            
            if df.empty:
                self._log("CRITICAL ERROR: Dataset is empty after merging/cleaning.")
                self._log("Hint: Your files might not have overlapping dates, or 'dropna' removed everything.")
                self._log("Try loading fewer files (e.g., just Price and Fees) to test.")
                return

            # 2. Reset Notebook
            try:
                for tab_id in self.notebook.tabs():
                    self.notebook.forget(tab_id)
                self._log("Notebook tabs cleared.")
            except Exception as e:
                self._log(f"Warning clearing tabs: {e}")

            # 3. Define the Tab Creator with Error Handling
            def create_independent_plot_tab(title, plot_func):
                self._log(f"Generating plot: {title}...")
                try:
                    frame = tk.Frame(self.notebook, bg=THEME["bg"])
                    self.notebook.add(frame, text=title)
                    
                    # Setup Matplotlib Figure
                    fig = Figure(figsize=(10, 8), dpi=100) 
                    ax = fig.add_subplot(111)
                    
                    # Execute the specific plotting function
                    plot_func(ax)
                    
                    # Styling
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color=THEME["fg"])
                    for spine in ax.spines.values(): spine.set_color("#333")
                    
                    # Embed in Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas.draw()
                    
                    # Toolbar
                    toolbar = NavigationToolbar2Tk(canvas, frame)
                    toolbar.config(background=THEME["panel"])
                    toolbar._message_label.config(background=THEME["panel"], foreground="white")
                    for button in toolbar.winfo_children(): button.config(background=THEME["panel"])
                    toolbar.update()
                    
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    self._log(f" -> Success: {title}")
                    
                except Exception as e:
                    self._log(f" -> FAILED {title}: {str(e)}")
                    import traceback
                    traceback.print_exc() # Print full error to terminal console

            # --- PLOT DEFINITIONS (Safe Versions) ---

            def p_regime_scatter(ax):
                norm = df[df['Regime']=='NORMAL']
                shock = df[df['Regime']=='SHOCK']
                if len(norm) > 0: ax.scatter(norm['TCI'], norm['Vel_30d_Change'], s=15, color=THEME["safe"], alpha=0.3, label="Normal")
                if len(shock) > 0: ax.scatter(shock['TCI'], shock['Vel_30d_Change'], s=30, color=THEME["shock"], alpha=0.8, label="Shock")
                ax.axvline(thresh, color="white", linestyle="--", alpha=0.5, label="Horizon")
                ax.axhline(0, color="white", linestyle="-", alpha=0.1)
                ax.set_xlabel("Friction (TCI)")
                ax.set_ylabel("30-Day Velocity Impact (%)")
                ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_prob_collapse(ax):
                v_norm = df[df['Regime']=='NORMAL']['Vel_30d_Change'].dropna()
                v_shock = df[df['Regime']=='SHOCK']['Vel_30d_Change'].dropna()
                if len(v_norm) > 1 and len(v_shock) > 1:
                    sns.kdeplot(v_norm, ax=ax, color=THEME["safe"], fill=True, alpha=0.2, label="Normal")
                    sns.kdeplot(v_shock, ax=ax, color=THEME["shock"], fill=True, alpha=0.2, label="Shock")
                    ax.set_xlabel("Velocity Growth Probability")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")
                else: ax.text(0.5, 0.5, "Insufficient Data for PDF", ha='center', color='yellow')

            def p_damage_box(ax):
                data = [df[df['Regime']=='NORMAL']['Vel_30d_Change'].dropna(),
                        df[df['Regime']=='SHOCK']['Vel_30d_Change'].dropna()]
                if len(data[0]) > 0 and len(data[1]) > 0:
                    bp = ax.boxplot(data, patch_artist=True, tick_labels=['Normal', 'Shock'], widths=0.5)
                    colors = [THEME["safe"], THEME["shock"]]
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    for median in bp['medians']: median.set_color('white')
                    ax.set_ylabel("30-Day Velocity Impact (%)")
                else: ax.text(0.5, 0.5, "Insufficient Data for Box Plot", ha='center', color='yellow')

            def p_timeline(ax):
                ax.plot(df.index, df['Velocity'], color=THEME["accent"], lw=1, label="Robust Velocity")
                ylim = ax.get_ylim()
                ax.fill_between(df.index, ylim[0], ylim[1], where=(df['Regime']=='SHOCK'), color=THEME["shock"], alpha=0.3)
                ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_bootstrap(ax):
                # FIX: Handle zeros for log scale to prevent crash
                plot_data = df['Insolvency'].replace(0, 0.0001)
                ax.plot(df.index, plot_data, color=THEME["warn"], lw=1)
                ax.axhline(self.engine.insol_thresh, color=THEME["shock"], linestyle="--", label=f"Insolvency ({self.engine.insol_lbl})")
                ax.set_yscale('log')
                ax.set_ylabel(f"Cost ({self.engine.insol_lbl})")
                ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_divergence(ax):
                if 'Div_Score' in df.columns:
                    ax.fill_between(df.index, 0, df['Div_Score'], where=(df['Div_Score'] > 0), color=THEME["shock"], alpha=0.8, label="Artificial Vol")
                    ax.fill_between(df.index, 0, df['Div_Score'], where=(df['Div_Score'] <= 0), color=THEME["safe"], alpha=0.3, label="Organic")
                    ax.set_ylabel("Z-Score Divergence")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")
                else: ax.text(0.5, 0.5, "Volume/Organic Data Missing", ha='center')

            def p_hysteresis(ax):
                # FIX: Drop NaNs from rolling window so color array matches geometry length
                df_plot = df[['TCI_Log', 'Vel_Log']].rolling(14).mean().dropna()
                if not df_plot.empty:
                    x = df_plot['TCI_Log']
                    y = df_plot['Vel_Log']
                    points = ax.scatter(x, y, c=range(len(x)), cmap='twilight', s=15, alpha=0.8)
                    ax.set_xlabel("Log(Friction)")
                    ax.set_ylabel("Log(Utility)")
                    cbar = ax.figure.colorbar(points, ax=ax)
                    cbar.set_label("Time")
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                else:
                    ax.text(0.5, 0.5, "Insufficient Data for Hysteresis", ha='center', color='yellow')

            def p_stagflation(ax):
                colors = {'Boom':THEME["safe"], 'Speculation':THEME["warn"], 'Stagflation':"#b00b69", 'Capitulation':THEME["shock"]}
                found = False
                for state, col in colors.items():
                    sub = df[df['State']==state]
                    if not sub.empty:
                        found = True
                        ax.scatter(sub['Price_Mom'], sub['Vel_Mom'], c=col, s=15, label=state, alpha=0.6)
                if found:
                    ax.axhline(0, c='white', alpha=0.3, ls='--')
                    ax.axvline(0, c='white', alpha=0.3, ls='--')
                    # FIX: Added Labels
                    ax.set_xlabel("Price Momentum (30d %)")
                    ax.set_ylabel("Utility Momentum (30d %)")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")
                else: ax.text(0.5, 0.5, "No State Data", ha='center')

            def p_sensitivity(ax):
                pcts = range(80, 100)
                net_damage = []
                for p in pcts:
                    val = df['TCI'].quantile(p/100)
                    s_mean = df[df['TCI'] > val]['Vel_30d_Change'].mean()
                    n_mean = df[df['TCI'] <= val]['Vel_30d_Change'].mean()
                    net_damage.append(s_mean - n_mean)
                ax.plot(pcts, net_damage, color=THEME["shock"], marker='o')
                ax.axhline(0, color='white', linestyle='--')
                ax.set_xlabel("Threshold Percentile")
                ax.set_ylabel("Net Damage (%)")

            def p_event_horizon(ax):
                # Hexbin Density Plot
                if 'Mempool' in df.columns and 'Delay' in df.columns:
                    subset = df[(df['Mempool'] > 0) & (df['Delay'] > 0)]
                    if len(subset) > 10:
                        hb = ax.hexbin(subset['Mempool'], subset['Delay'], gridsize=25, cmap='inferno', mincnt=1, bins='log', linewidths=0)
                        ax.set_xlabel("Mempool (Bytes)")
                        ax.set_ylabel("Delay (Min)")
                        cb = ax.figure.colorbar(hb, ax=ax)
                        cb.set_label("Density (Log)")
                        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                        horizon_x = subset['Mempool'].quantile(0.90)
                        ax.axvline(horizon_x, color=THEME["safe"], linestyle="--", alpha=0.5)
                        ax.text(horizon_x, subset['Delay'].max(), " Saturation", color=THEME["safe"], fontsize=8)
                        ax.grid(False)
                    else: ax.text(0.5, 0.5, "Insufficient Data (>0)", ha='center', color='yellow')
                else: ax.text(0.5, 0.5, "Mempool/Delay Missing", ha='center', color='red')
            
            def p_utxo_health(ax):
                if 'UTXO' in df.columns:
                    ax.plot(df.index, df['UTXO'], color=THEME["fg"], lw=1, label="UTXO Set Size")
                    
                    # Highlight Bloat (Post-2023 Spam)
                    ylim = ax.get_ylim()
                    if 'State_Bloat' in df.columns:
                         ax.fill_between(df.index, ylim[0], ylim[1], where=(df['State_Bloat']==1), color=THEME["l2"], alpha=0.3, label="State Bloat (Ordinals)")
                    
                    # Highlight Exodus (Retail Consolidation)
                    if 'Confirmed_Exodus' in df.columns:
                         ax.fill_between(df.index, ylim[0], ylim[1], where=(df['Confirmed_Exodus']==1), color=THEME["shock"], alpha=0.5, label="Retail Exodus")
                    
                    ax.set_ylabel("Unspent Output Count")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")
                else:
                    ax.text(0.5, 0.5, "UTXO Data Missing", ha='center', color='red')

            # 4. Execute Creation Loop
            self._log("--- Building Tabs ---")
            create_independent_plot_tab("Regime Scatter", p_regime_scatter)
            create_independent_plot_tab("Probability Collapse", p_prob_collapse)
            create_independent_plot_tab("Damage Assessment", p_damage_box)
            create_independent_plot_tab("Shock Timeline", p_timeline)
            create_independent_plot_tab("Insolvency Paradox", p_bootstrap)
            create_independent_plot_tab("Whale Divergence", p_divergence)
            create_independent_plot_tab("UTXO Structural Health", p_utxo_health)
            create_independent_plot_tab("Phase Hysteresis", p_hysteresis)
            create_independent_plot_tab("Stagflation Matrix", p_stagflation)
            create_independent_plot_tab("Sensitivity Curve", p_sensitivity)
            create_independent_plot_tab("Event Horizon", p_event_horizon)
            
            self._log("--- VISUALIZATION COMPLETE ---")
            
    def _export(self):
            if self.engine is None: return
            self._log("Generating Full Forensic Report...")
            df = self.engine.df.copy()

            # ==============================================================================
            #  1. AGGREGATION ENGINE (Event Grouping)
            # ==============================================================================
            shock_subset = df[df['Regime'] == 'SHOCK'].sort_index()
            events_data = []
            
            if not shock_subset.empty:
                # Group consecutive shock days into single events
                shock_subset['grp'] = (shock_subset.index.to_series().diff().dt.days > 1).cumsum()
                
                for _, group in shock_subset.groupby('grp'):
                    # Calculate per-event metrics
                    start, end = group.index.min(), group.index.max()
                    duration = (end - start).days + 1
                    
                    # Safe pct change calc
                    p_start, p_end = group.iloc[0]['Price'], group.iloc[-1]['Price']
                    v_start, v_end = group.iloc[0]['Velocity'], group.iloc[-1]['Velocity']
                    price_imp = ((p_end - p_start) / p_start) * 100
                    vel_imp = ((v_end - v_start) / v_start) * 100
                    
                    dom_state = group['State'].mode()[0] if not group['State'].mode().empty else "Unstable"
                    
                    events_data.append({
                        'Start': start, 'End': end, 'Duration': duration,
                        'Peak_TCI': group['TCI'].max(),
                        'Avg_Fee': group['Fees'].mean(),
                        'Avg_Delay': group['Delay'].mean(),
                        'Price_Imp': price_imp,
                        'Vel_Imp': vel_imp,
                        'State': dom_state
                    })
            
            events_df = pd.DataFrame(events_data).sort_values('Peak_TCI', ascending=False) if events_data else pd.DataFrame()

            # ==============================================================================
            #  2. STATISTICS & RISK
            # ==============================================================================
            # Regime Comparison
            regime_stats = df.groupby('Regime')[['Fees', 'Delay', 'TCI', 'Velocity', 'Vel_30d_Change']].mean()
            
            # Insolvency Stats
            insol_df = df[df['Insolvency'] > self.engine.insol_thresh]
            insol_stats = {
                'days': len(insol_df),
                'max': df['Insolvency'].max(),
                'stag_prob': (shock_subset['State'].value_counts().get('Stagflation', 0) / len(shock_subset) * 100) if not shock_subset.empty else 0
            }

            # Granger Causality (Volume vs Fees)
            granger_html = ""
            if STATS_AVAILABLE and 'Organic' in df.columns:
                try:
                    d_diff = df[['Organic', 'Fees']].pct_change().dropna().replace([np.inf, -np.inf], 0)
                    res = grangercausalitytests(d_diff, maxlag=[14], verbose=False)
                    p_val = res[14][0]['ssr_ftest'][1]
                    color = "#ff003c" if p_val < 0.05 else "#888"
                    verdict = "CONFIRMED" if p_val < 0.05 else "UNPROVEN"
                    # FIX: Merged columns to match HTML header (Metric | Value | Implication)
                    granger_html = f"<tr><td><strong>Granger Causality (Volume)</strong></td><td class='mono'>Lag 14 | p={p_val:.4f}</td><td style='color:{color}; font-weight:bold;'>{verdict}</td></tr>"
                except: pass


            # ==============================================================================
            #  3. MATH VALIDATION (Running the Validator)
            # ==============================================================================
            validation = {
                'elasticity': EconometricValidator.run_log_log_elasticity(df),
                'threshold': EconometricValidator.run_threshold_regression(df),
                'welch': EconometricValidator.run_welchs_t_test(df),
                'bootstrap': EconometricValidator.run_bootstrap_ci(df)
            }

            # ==============================================================================
            #  4. EXPORT
            # ==============================================================================
            html = ReportGenerator.generate(self.engine, events_df, regime_stats, insol_stats, validation, granger_html)
            
            fname = f"Full_Forensic_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            with open(fname, "w", encoding="utf-8") as f: 
                f.write(html)
                
            webbrowser.open(f"file://{os.path.abspath(fname)}")
            self._log(f"Full Detailed Report Exported: {fname}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ResearchSuite(root)
    root.mainloop()
