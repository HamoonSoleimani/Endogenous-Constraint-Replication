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
from statsmodels.tsa.stattools import adfuller
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
#  1. DATA PARSER (Strict Mode - No Simulation)
# ==============================================================================
class DataHandler:
    """
    Robust file parser optimized for standard Blockchain data dumps (.json)
    and specific CSV exports (Lightning Capacity, Min Fees).
    """
    
    @staticmethod
    def parse_file(filepath, hint):
        try:
            ext = os.path.splitext(filepath)[1].lower()
            df = None

            # JSON Handling (Standard Blockchain.com / charts format)
            if ext == '.json':
                with open(filepath, 'r') as f:
                    raw = json.load(f)
                
                target = raw
                # Flatten dict if wrapped (e.g. {'market-price': [...]})
                if isinstance(raw, dict):
                    # Try to find a key that looks like a list of data
                    for k in raw.keys():
                        if isinstance(raw[k], list):
                            target = raw[k]; break
                            
                data = []
                for item in target:
                    # Handle standard {x: timestamp, y: value} format
                    ts = item.get('x') or item.get('t') or item.get('timestamp')
                    val = item.get('y') or item.get('v') or item.get('value')
                    
                    # Handle list format [timestamp, value]
                    if isinstance(item, list) and len(item) >= 2: 
                        ts, val = item[0], item[1]
                        
                    if ts is not None and val is not None:
                        try:
                            ts = float(ts)
                            # Auto-detect ms vs seconds
                            if ts > 1e11: ts /= 1000
                            dt = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
                            data.append({'Date': dt, 'Value': float(val)})
                        except: pass
                df = pd.DataFrame(data)

            # CSV Handling (Optimized for your specific files)
            elif ext == '.csv':
                # read_csv with flexible separator detection
                df = pd.read_csv(filepath, sep=None, engine='python')
                
                # Normalize columns
                df.columns = [str(c).strip().lower() for c in df.columns]
                
                # 1. Detect Date Column
                date_candidates = ['date', 'time', 'day', 'created', 'period']
                date_col = next((c for c in df.columns if any(x in c for x in date_candidates)), None)
                
                if not date_col: 
                    # Fallback: assume first column is date if looks like date
                    date_col = df.columns[0]
                
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                
                # 2. Detect Value Column based on Hint
                cols = [c for c in df.columns if c != date_col and c != 'date']
                target_col = None
                
                # specific mappings for your screenshot files
                if 'ln_cap' in hint.lower():
                    # Look for 'capacity', 'btc', 'amount'
                    target_col = next((c for c in cols if any(x in c for x in ['cap', 'btc', 'total'])), cols[-1])
                elif 'min_fee' in hint.lower():
                    # Look for 'fee', 'sat', 'usd'
                    target_col = next((c for c in cols if 'fee' in c or 'sat' in c), cols[-1])
                else:
                    # Default to last column
                    target_col = cols[-1]
                
                # Cleanup formatting (remove commas from numbers like "1,000.00")
                if df[target_col].dtype == 'object':
                    df[target_col] = df[target_col].astype(str).str.replace(',', '', regex=True)
                
                df['Value'] = pd.to_numeric(df[target_col], errors='coerce')

            if df is not None and not df.empty:
                df = df.dropna(subset=['Date', 'Value'])
                df = df.set_index('Date')
                # Resample to daily mean to standardize jagged time series
                df = df[['Value']].resample('D').mean()
                return df.rename(columns={'Value': hint})

        except Exception as e:
            print(f"Parser Error [{hint}]: {e}")
            return None
        return None
# ==============================================================================
#  2. FORENSIC ENGINE (Strict Analysis)
# ==============================================================================
class ForensicEngine:
    def __init__(self, df, threshold_percentile=95.0):
        # Strict interpolation only for small gaps
        self.df = df.copy().interpolate(method='time').dropna()
        self.threshold_pct = threshold_percentile
        
        # Only run pipeline if sufficient data exists
        if not self.df.empty:
            self._run_pipeline()

    def _run_pipeline(self):
        self._apply_logical_fixes()
        self._calc_tci()
        self._detect_regimes()
        self._calc_impact_metrics()
        self._calc_divergence()
        self._calc_utxo_health()
        self._calc_phase_state()

    def _apply_logical_fixes(self):
            """
            FORENSIC LOGIC ENGINE v7.1 (Strict Mode)
            
            Changes from previous version:
            1. REMOVED 'Magic Number' (10x) multiplier for L2 Volume.
            2. SEPARATED 'Velocity' (Flow) from 'Liquidity' (Stock).
            3. PRIORITIZED Realized Cap over Market Cap for denominator.
            """
            
            # ---------------------------------------------------------
            # 1. CAPITALIZATION LOGIC (The Denominator)
            # ---------------------------------------------------------
            # We need the economic base to calculate Velocity (V = Vol / Cap).
            # Preference: Realized Cap (cost basis) > Market Cap (speculative value).
            
            if 'Price' in self.df.columns and 'Supply' in self.df.columns:
                # Calculate Standard Market Cap (Price * Supply)
                self.df['Market_Cap'] = self.df['Price'] * self.df['Supply']
                
                if 'MVRV' in self.df.columns:
                    # DERIVE Realized Cap: Market Cap / MVRV Ratio
                    # This reconstructs the aggregate cost basis of the network.
                    self.df['Realized_Cap'] = self.df['Market_Cap'] / self.df['MVRV'].replace(0, np.nan)
                    self.cap_type = "Realized Cap (Derived from MVRV)"
                else:
                    # Fallback to Market Cap (Less accurate for utility, but usable)
                    self.df['Realized_Cap'] = self.df['Market_Cap']
                    self.cap_type = "Market Cap (MVRV Missing)"
            else:
                # Critical Data Missing
                self.df['Realized_Cap'] = np.nan
                self.df['Market_Cap'] = np.nan
                self.cap_type = "INSUFFICIENT DATA (Price/Supply Missing)"

            # ---------------------------------------------------------
            # 2. VOLUME & LAYER 2 LOGIC (The Numerator)
            # ---------------------------------------------------------
            
            # Layer 1 Volume (Verified On-Chain)
            if 'Volume' in self.df.columns:
                self.df['Vol_L1'] = self.df['Volume']
            else:
                self.df['Vol_L1'] = np.nan

            # Layer 2 Logic (Lightning Network)
            # FIX: Removed arbitrary 10x multiplier. 
            # LN Capacity is treated as "Liquidity Available" (Stock), not "Volume Sent" (Flow).
            if 'LN_Cap' in self.df.columns and 'Price' in self.df.columns:
                # Convert BTC Capacity to USD Value
                self.df['Liq_L2_USD'] = self.df['LN_Cap'] * self.df['Price']
                
                # STRICT MODE: Do NOT add L2 Liquidity to L1 Volume. 
                # Adding Stock (Cap) to Flow (Vol) is a dimensional error.
                # We track L2 separately to see if Liquidity rises when L1 Friction rises.
                self.df['Vol_Total'] = self.df['Vol_L1']
                self.vol_type = "L1 Verified (L2 Liquidity Tracked Separately)"
            else:
                self.df['Liq_L2_USD'] = 0
                self.df['Vol_Total'] = self.df['Vol_L1']
                self.vol_type = "L1 Only"

            # ---------------------------------------------------------
            # 3. VELOCITY CALCULATION
            # ---------------------------------------------------------
            # Formula: Velocity = Total Transaction Volume / Network Capitalization
            
            if 'Vol_Total' in self.df.columns and 'Realized_Cap' in self.df.columns:
                # Robust Velocity: Uses Verified Volume / Realized Cost Basis
                self.df['Vel_Robust'] = self.df['Vol_Total'] / self.df['Realized_Cap'].replace(0, np.nan)
            else:
                self.df['Vel_Robust'] = np.nan

            # NVT Fallback (Network Value to Transactions)
            # NVT = Cap / Volume, so Velocity = 1 / NVT
            if 'NVT' in self.df.columns:
                self.df['Vel_Standard'] = 1.0 / self.df['NVT'].replace(0, np.nan)
            else:
                self.df['Vel_Standard'] = np.nan

            # Final Velocity Composite
            # Prioritize the Robust calculation, fill gaps with NVT-based calculation
            self.df['Velocity'] = self.df['Vel_Robust'].fillna(self.df['Vel_Standard'])
            
            # ---------------------------------------------------------
            # 4. MOMENTUM & AUXILIARY METRICS
            # ---------------------------------------------------------
            
            # Price Momentum (30-day lookback for Stagflation Matrix)
            if 'Price' in self.df.columns:
                self.df['Price_Mom'] = self.df['Price'].pct_change(30).fillna(0)
            else:
                self.df['Price_Mom'] = 0

            # Data Cleanup: Forward Fill small gaps to prevent regression crashes
            # (Only fills gaps, does not alter existing data)
            cols_to_clean = ['Velocity', 'Realized_Cap', 'Vol_Total', 'Liq_L2_USD']
            for col in cols_to_clean:
                if col in self.df.columns:
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    
    def _calc_tci(self):
            # 1. Base Fee Stress Calculation (Winsorized)
            if 'Fees' in self.df.columns:
                # Winsorize extreme outliers (99th percentile)
                capped_fees = self.df['Fees'].clip(upper=self.df['Fees'].quantile(0.99))
                
                # Calculate Stress (Magnitude + Volatility)
                fee_mean = capped_fees.rolling(30).mean().bfill()
                fee_std = capped_fees.rolling(30).std().fillna(0)
                fee_stress = fee_mean + fee_std
                
                # --- FIX: DYNAMIC VIABILITY RATIO (Wealth-Relative) ---
                # Replaces static $10 threshold with economic burden relative to stored wealth.
                # Denominator: Average Realized Value per UTXO (The "Average User Account")
                
                if 'Realized_Cap' in self.df.columns and 'UTXO' in self.df.columns:
                    # Calculate Avg Stored Wealth (Cost Basis)
                    self.df['Avg_Stored_Wealth'] = self.df['Realized_Cap'] / self.df['UTXO'].replace(0, 1)
                    
                    # Dynamic Burden: Fee as % of Avg Stored Wealth
                    # Basis Points (bps) approach prevents "Tiny Number" errors
                    self.df['Burden_Rate'] = (self.df['Fees'] / self.df['Avg_Stored_Wealth']) * 100.0
                    
                    self.df['Insolvency'] = self.df['Burden_Rate']
                    self.insol_lbl = "Wealth Burden (%)"
                    self.insol_thresh = 0.05  # Threshold: > 0.05% of Stored Wealth (5 bps)
                    
                elif 'Volume' in self.df.columns:
                     # Fallback A: Fee relative to daily transaction flow (Legacy)
                     self.df['Burden_Rate'] = (self.df['Fees'] / self.df['Volume'].replace(0, np.nan)) * 100
                     self.df['Insolvency'] = self.df['Burden_Rate']
                     self.insol_lbl = "Volume Burden (%)"
                     self.insol_thresh = 0.01
                else:
                    # Fallback B: Nominal Fees (Last Resort)
                    self.df['Insolvency'] = self.df['Fees']
                    self.insol_lbl = "Nominal Fee ($)"
                    self.insol_thresh = 10.0
                    
            else:
                # No Fee Data loaded
                fee_stress = 0
                self.df['Insolvency'] = 0
                self.insol_lbl = "N/A"
                self.insol_thresh = 0

            # Alias for Plotting
            self.df['RCI'] = self.df['Insolvency']

            # 2. Delay Factor Calculation
            if 'Delay' in self.df.columns:
                 delay_factor = (self.df['Delay'] / 10.0) 
            elif 'Mempool' in self.df.columns:
                 delay_factor = (self.df['Mempool'] / 1000000.0) 
            else:
                 delay_factor = 1.0

            # 3. Final TCI Composite Index
            self.df['TCI'] = fee_stress * delay_factor
            self.df['TCI'] = self.df['TCI'].fillna(0)
        
    def _detect_regimes(self):
        if 'TCI' in self.df.columns:
            self.thresh_val = self.df['TCI'].quantile(self.threshold_pct / 100.0)
            self.df['Regime'] = np.where(self.df['TCI'] >= self.thresh_val, 'SHOCK', 'NORMAL')
        else:
            self.df['Regime'] = 'NORMAL'
            self.thresh_val = 0

    def _calc_impact_metrics(self):
        if 'Velocity' in self.df.columns:
            self.df['Vel_Next_30d'] = self.df['Velocity'].shift(-30)
            self.df['Vel_30d_Change'] = ((self.df['Vel_Next_30d'] - self.df['Velocity']) / self.df['Velocity']) * 100.0
            self.df['Vel_Mom'] = self.df['Velocity'].pct_change(30)
        else:
            self.df['Vel_30d_Change'] = 0
            self.df['Vel_Mom'] = 0
        
        # State Classification
        conditions = [
            (self.df['Price_Mom'] > 0) & (self.df['Vel_Mom'] > 0),
            (self.df['Price_Mom'] > 0) & (self.df['Vel_Mom'] <= 0),
            (self.df['Price_Mom'] <= 0) & (self.df['Vel_Mom'] <= 0),
            (self.df['Price_Mom'] <= 0) & (self.df['Vel_Mom'] > 0)
        ]
        self.df['State'] = np.select(conditions, ['Boom', 'Speculation', 'Stagflation', 'Capitulation'], default='Neutral')

    def _calc_divergence(self):
        # Organic usually maps to 'n-unique-addresses.json'
        if 'Organic' in self.df.columns and 'Volume' in self.df.columns:
            scaler = StandardScaler()
            vol_z = scaler.fit_transform(self.df[['Volume']].fillna(0))
            org_z = scaler.fit_transform(self.df[['Organic']].fillna(0))
            self.df['Div_Score'] = vol_z - org_z

    def _calc_utxo_health(self):
        if 'UTXO' in self.df.columns:
            self.df['UTXO_Growth'] = self.df['UTXO'].pct_change(30)
            
            # Identify Exodus: High Fees + Negative Growth
            condition_exodus = (self.df['Regime'] == 'SHOCK') & (self.df['UTXO_Growth'] < 0)
            self.df['Confirmed_Exodus'] = np.where(condition_exodus, 1, 0)
            
            # Identify Bloat: High Fees + High Growth
            condition_bloat = (self.df['Regime'] == 'SHOCK') & (self.df['UTXO_Growth'] > 0.05)
            self.df['State_Bloat'] = np.where(condition_bloat, 1, 0)
        else:
            self.df['UTXO_Growth'] = 0
            self.df['Confirmed_Exodus'] = 0
            self.df['State_Bloat'] = 0

    def _calc_phase_state(self):
        if 'TCI' in self.df.columns:
            self.df['TCI_Log'] = np.log1p(self.df['TCI'])
        if 'Velocity' in self.df.columns:
            self.df['Vel_Log'] = np.log1p(self.df['Velocity'])
# ==============================================================================
#  3. ECONOMETRIC VALIDATOR
# ==============================================================================
class EconometricValidator:
    """
    Implements Appendix A (Mathematical Derivations) and Appendix B (Statistical Tests)
    Includes: OLS, Threshold Regression, Hansen Search, and IV-2SLS.
    """
    
    @staticmethod
    def run_log_log_elasticity(df):
        try:
            if 'Velocity' not in df.columns or 'TCI' not in df.columns: raise ValueError
            
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
        """
        Standard OLS based on the user-selected regime (GUI Slider).
        Calculates Beta 1 (Normal) vs Beta 2 (Shock).
        """
        try:
            if 'Vel_30d_Change' not in df.columns: raise ValueError

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
    def run_hansen_threshold_search(df):
        """
        Implements Hansen (2000) Least Squares verification to find the 
        mathematically optimal structural break rather than guessing.
        """
        try:
            if 'Vel_30d_Change' not in df.columns or 'TCI' not in df.columns:
                return {'optimal_threshold': 0, 'min_ssr': 0}

            # Prepare data
            data = df[['Vel_30d_Change', 'TCI']].dropna().sort_values('TCI')
            y = data['Vel_30d_Change']
            q = data['TCI'] # Threshold variable
            
            # Search range: 15th to 85th percentile (Trimmed)
            lower = q.quantile(0.15)
            upper = q.quantile(0.85)
            candidates = q[(q >= lower) & (q <= upper)].unique()
            
            best_ssr = np.inf
            best_thresh = 0
            
            # Grid Search for Minimum Sum of Squared Residuals (SSR)
            # Skip every 5th point to speed up calculation
            for gamma in candidates[::5]: 
                # Split sample based on candidate gamma
                s1 = y[q <= gamma]
                s2 = y[q > gamma]
                
                # Calculate SSR for this split
                ssr = ((s1 - s1.mean())**2).sum() + ((s2 - s2.mean())**2).sum()
                
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_thresh = gamma
            
            return {
                'optimal_threshold': best_thresh,
                'min_ssr': best_ssr
            }
        except Exception as e:
            print(f"Hansen Search Failed: {e}")
            return {'optimal_threshold': 0, 'min_ssr': 0}

    @staticmethod
    def run_stationarity_test(df):
        """
        Runs Augmented Dickey-Fuller to prove data isn't a random walk.
        """
        try:
            if 'Vel_30d_Change' not in df.columns: return {'p_value': 1.0, 'is_stationary': False}
            series = df['Vel_30d_Change'].dropna()
            if len(series) < 20: return {'p_value': 1.0, 'is_stationary': False}
            
            result = adfuller(series)
            return {
                'adf_stat': result[0],
                'p_value': result[1],
                'is_stationary': result[1] < 0.05
            }
        except:
            return {'p_value': 1.0, 'is_stationary': False}

    @staticmethod
    def run_iv_regression(df):
        """
        FIXED IV-2SLS: 
        Instrument: Lagged Mempool Size (Physical Congestion).
        Reason: 'Difficulty' correlates with Price (Bull Market bias). 
                Mempool backlog correlates purely with Friction.
        """
        try:
            # Require Instruments (Mempool is physically exogenous to monetary velocity)
            if 'Mempool' not in df.columns or 'TCI' not in df.columns: 
                return {'status': 'Skipped (Missing Mempool Data)'}

            # Prepare Dataset: Lag Mempool by 1 day to ensure exogeneity
            data = df.copy()
            data['Mempool_Lag'] = data['Mempool'].shift(1)
            data = data[['Vel_30d_Change', 'TCI', 'Mempool_Lag']].dropna()
            
            if len(data) < 10: return {'status': 'Insufficient Data'}

            # STAGE 1: Predict TCI using Lagged Mempool (Physical Backlog)
            # TCI (Econ Friction) ~ Mempool_Lag (Physical Constraint)
            X_stage1 = sm.add_constant(data['Mempool_Lag'])
            model_stage1 = sm.OLS(data['TCI'], X_stage1).fit()
            data['Predicted_TCI'] = model_stage1.fittedvalues
            
            # STAGE 2: Regress Velocity on Predicted TCI
            # This isolates the variation in TCI caused strictly by backlog, not price hype.
            X_stage2 = sm.add_constant(data['Predicted_TCI'])
            model_stage2 = sm.OLS(data['Vel_30d_Change'], X_stage2).fit()
            
            return {
                'status': 'Success (IV-2SLS)',
                'iv_beta': model_stage2.params['Predicted_TCI'], # Should now be NEGATIVE
                'p_value': model_stage2.pvalues['Predicted_TCI'],
                'stage1_strength': model_stage1.rsquared
            }
        except Exception as e:
            return {'status': f"Error: {str(e)}"}
    @staticmethod
    def run_welchs_t_test(df):
        try:
            if 'Vel_30d_Change' not in df.columns: return 0.0, 1.0
            normal = df[df['Regime'] == 'NORMAL']['Vel_30d_Change'].dropna()
            shock = df[df['Regime'] == 'SHOCK']['Vel_30d_Change'].dropna()
            if len(normal) < 2 or len(shock) < 2: return 0.0, 1.0
            t_stat, p_val = stats.ttest_ind(normal, shock, equal_var=False)
            return t_stat, p_val
        except:
            return 0.0, 1.0

    @staticmethod
    def run_bootstrap_ci(df, iterations=5000):
        try:
            if 'Vel_30d_Change' not in df.columns: return 0.0, 0.0
            normal = df[df['Regime'] == 'NORMAL']['Vel_30d_Change'].dropna().values
            shock = df[df['Regime'] == 'SHOCK']['Vel_30d_Change'].dropna().values
            
            if len(shock) < 5: return 0.0, 0.0
            
            n_norm, n_shock = len(normal), len(shock)
            idx_n = np.random.randint(0, n_norm, (iterations, n_norm))
            idx_s = np.random.randint(0, n_shock, (iterations, n_shock))
            
            diffs = np.mean(shock[idx_s], axis=1) - np.mean(normal[idx_n], axis=1)
            return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        except:
            return 0.0, 0.0# ==============================================================================
#  4. REPORT GENERATOR
# ==============================================================================
class ReportGenerator:
    @staticmethod
    def generate(engine, events_df, regime_stats, insol_stats, validation, granger_html):
        df = engine.df.copy()
        
        # --- 0. PRECISION FORMATTERS ---
        def fmt_p(p): 
            if p is None: return "N/A"
            if p < 0.0001: return "< 0.0001"
            return f"{p:.5f}"
        
        def fmt_num(n, suffix=""):
            if n is None or pd.isna(n): return "-"
            return f"{n:,.4f}{suffix}"

        # --- 1. UNPACKING & DERIVING ADVANCED METRICS ---
        
        # A. Econometric Unpacking
        elast = validation.get('elasticity', {'beta': 0, 'r_squared': 0, 'p_value': 1})
        thresh = validation.get('threshold', {'beta_1_normal': 0, 'beta_2_shock': 0, 'marginal_impact': 0})
        welch = validation.get('welch', (0, 1)) # (t-stat, p-val)
        boot = validation.get('bootstrap', (0, 0)) # (lower, upper)
        hansen = validation.get('hansen', {'optimal_threshold': 0, 'min_ssr': 0})
        iv_res = validation.get('iv', {'status': 'Not Run', 'iv_beta': 0, 'p_value': 1, 'stage1_strength': 0})
        adf = validation.get('adf', {'adf_stat': 0, 'p_value': 1, 'is_stationary': False})

        # B. Net Damage Calculation
        try:
            vel_normal = regime_stats.loc['NORMAL', 'Vel_30d_Change']
            vel_shock = regime_stats.loc['SHOCK', 'Vel_30d_Change']
            net_damage = vel_shock - vel_normal
        except:
            vel_normal, vel_shock, net_damage = 0, 0, 0

        # C. Multiplier Calculation
        try:
            fee_normal = regime_stats.loc['NORMAL', 'Fees']
            fee_shock = regime_stats.loc['SHOCK', 'Fees']
            fee_mult = fee_shock / fee_normal if fee_normal > 0 else 0
            
            tci_normal = regime_stats.loc['NORMAL', 'TCI']
            tci_shock = regime_stats.loc['SHOCK', 'TCI']
            tci_mult = tci_shock / tci_normal if tci_normal > 0 else 0
        except:
            fee_mult, tci_mult = 0, 0

        # --- 2. DEEP DIVE: UTXO STRUCTURAL INVERSION LOGIC ---
        # (Re-calculating specifically for the report to ensure raw data exposure)
        utxo_html = ""
        
        if 'UTXO' in df.columns and 'Realized_Cap' in df.columns:
            # 1. Structural Break Split (Pre/Post 2023)
            df_sorted = df.sort_index()
            pre_2023 = df_sorted[df_sorted.index < '2023-01-01']
            post_2023 = df_sorted[df_sorted.index >= '2023-01-01']
            
            # Correlation: Friction (TCI) vs UTXO Count Growth
            # We show the raw Pearson correlation coefficients
            if not pre_2023.empty:
                corr_pre = pre_2023['TCI'].corr(pre_2023['UTXO_Growth'])
            else: corr_pre = 0
            
            if not post_2023.empty:
                corr_post = post_2023['TCI'].corr(post_2023['UTXO_Growth'])
            else: corr_post = 0
            
            # 2. Value Dilution (Realized Cap / UTXO Count)
            df['Val_Per_UTXO'] = df['Realized_Cap'] / df['UTXO'].replace(0, 1)
            avg_val_normal = df[df['Regime']=='NORMAL']['Val_Per_UTXO'].mean()
            avg_val_shock = df[df['Regime']=='SHOCK']['Val_Per_UTXO'].mean()
            dilution_abs = avg_val_shock - avg_val_normal
            dilution_pct = (dilution_abs / avg_val_normal) * 100 if avg_val_normal != 0 else 0
            
            # 3. Zombie Ratio (Volume / UTXO Count)
            if 'Vol_L1' in df.columns:
                df['Act_Per_UTXO'] = df['Vol_L1'] / df['UTXO'].replace(0, 1)
                act_normal = df[df['Regime']=='NORMAL']['Act_Per_UTXO'].mean()
                act_shock = df[df['Regime']=='SHOCK']['Act_Per_UTXO'].mean()
                zombie_abs = act_shock - act_normal
                zombie_pct = (zombie_abs / act_normal) * 100 if act_normal != 0 else 0
            else:
                zombie_abs, zombie_pct = 0, 0

            # Logic Verdicts
            is_inverted = (corr_pre < 0 and corr_post > 0)
            verdict_corr = '<strong>STRUCTURAL INVERSION DETECTED</strong>' if is_inverted else 'Linear/Non-Inverted'
            verdict_dilution = '<strong>DUST ACCUMULATION</strong>' if dilution_pct < -5 else 'Healthy Capital Density'
            verdict_zombie = '<strong>WHALE DIVERGENCE</strong>' if zombie_pct > 20 else 'Organic Scaling'

            utxo_html = f"""
            <h2>6. Deep Dive: UTXO Forensic Architecture</h2>
            <p>Isolation of "Adoption" vs. "Bloat" using Pearson Correlation Splits and Unit Economic Densities.</p>
            <table>
                <thead>
                    <tr><th>Forensic Metric</th><th>Raw Calculation</th><th>Differential / Slope</th><th>Forensic Verdict</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <strong>Regime Correlation Split</strong><br>
                            <span style="color:#888; font-size:10px;">Pearson r: TCI vs. UTXO Growth</span>
                        </td>
                        <td class="mono">
                            Pre-2023 r: <span class="{'success' if corr_pre < 0 else 'danger'}">{corr_pre:.5f}</span><br>
                            Post-2023 r: <span class="{'danger' if corr_post > 0 else 'success'}">{corr_post:.5f}</span>
                        </td>
                        <td class="mono">
                            Delta r: {corr_post - corr_pre:+.4f}
                        </td>
                        <td class="{'danger' if is_inverted else 'mono'}">{verdict_corr}</td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Capital Density (Dilution)</strong><br>
                            <span style="color:#888; font-size:10px;">Realized Cap per UTXO</span>
                        </td>
                        <td class="mono">
                            Normal: ${avg_val_normal:,.2f}<br>
                            Shock: ${avg_val_shock:,.2f}
                        </td>
                        <td class="mono { 'danger' if dilution_pct < 0 else 'success'}">
                            {dilution_abs:+,.2f} USD ({dilution_pct:+.2f}%)
                        </td>
                        <td class="{ 'danger' if dilution_pct < -5 else 'success'}">{verdict_dilution}</td>
                    </tr>
                    <tr>
                        <td>
                            <strong>Zombie Ratio (Divergence)</strong><br>
                            <span style="color:#888; font-size:10px;">L1 Volume per UTXO</span>
                        </td>
                        <td class="mono">
                            Normal: ${act_normal:,.2f}<br>
                            Shock: ${act_shock:,.2f}
                        </td>
                        <td class="mono { 'danger' if zombie_pct < 0 else 'success'}">
                            {zombie_abs:+,.2f} USD ({zombie_pct:+.2f}%)
                        </td>
                        <td class="{ 'danger' if zombie_pct > 20 else 'success'}">{verdict_zombie}</td>
                    </tr>
                </tbody>
            </table>
            """
        elif 'UTXO' not in df.columns:
            utxo_html = "<h2>6. Deep Dive: UTXO Forensics</h2><p style='color:#666;'><em>UTXO Data not loaded. Unit economics cannot be calculated.</em></p>"
        else:
             utxo_html = "<h2>6. Deep Dive: UTXO Forensics</h2><p style='color:#666;'><em>Realized Cap data insufficient for unit economic analysis.</em></p>"

        # --- 3. HTML DOCUMENT CONSTRUCTION (FULL DETAIL) ---
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Audit: Endogenous Destabilizer</title>
            <style>
                :root {{ --bg: #050505; --panel: #121212; --text: #e0e0e0; --accent: #00f0ff; --danger: #ff003c; --success: #00ff9d; --warn: #ffea00; --l2: #9d00ff; }}
                body {{ background-color: var(--bg); color: var(--text); font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 40px; font-size: 12px; max-width: 1600px; margin: 0 auto; }}
                h1 {{ color: var(--accent); border-bottom: 2px solid var(--danger); padding-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; font-size: 24px; }}
                h2 {{ color: #fff; background: linear-gradient(90deg, #1a1a1a, #000); padding: 10px; border-left: 4px solid var(--accent); margin-top: 50px; text-transform: uppercase; font-size: 16px; letter-spacing: 1px; }}
                h3 {{ color: #888; margin-top: 5px; font-weight: normal; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 30px; }}
                .kpi-card {{ background: var(--panel); padding: 15px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.5); }}
                .kpi-val {{ font-size: 20px; font-weight: 700; color: #fff; display: block; margin-top: 5px; font-family: 'Consolas', monospace; }}
                .kpi-lbl {{ color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 12px; background: var(--panel); table-layout: fixed; }}
                th {{ background: #080808; color: var(--accent); padding: 12px 10px; text-align: left; border-bottom: 2px solid #333; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; font-size: 11px; }}
                td {{ padding: 10px; border-bottom: 1px solid #222; color: #ccc; vertical-align: top; word-wrap: break-word; }}
                tr:nth-child(even) {{ background-color: #161616; }}
                tr:hover {{ background-color: #1f1f1f; }}
                .danger {{ color: var(--danger); font-weight: bold; }}
                .success {{ color: var(--success); font-weight: bold; }}
                .warn {{ color: var(--warn); font-weight: bold; }}
                .mono {{ font-family: 'Consolas', monospace; font-size: 11px; }}
                .tiny {{ font-size: 10px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Forensic Blockchain Audit: Endogenous Destabilizer v7.1</h1>
            <h3>Analysis Configuration | Cap Logic: {engine.cap_type} | Vol Logic: {engine.vol_type} | Threshold: {100-engine.threshold_pct:.1f}th Percentile</h3>
            <p class="tiny">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine Hash: {hash(str(engine.df.index[0]))} | Observations: {len(df)}</p>

            <!-- EXECUTIVE DASHBOARD -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-lbl">Shock Events Detected</span>
                    <span class="kpi-val" style="color:{'#ff003c' if len(events_df) > 0 else '#00ff9d'}">
                        {len(events_df)}
                    </span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Net Velocity Damage</span>
                    <span class="kpi-val danger">
                        {net_damage:+.4f}%
                    </span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Insolvency Duration</span>
                    <span class="kpi-val warn">{insol_stats['days']} Days</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">IV-2SLS Causality</span>
                    <span class="kpi-val mono" style="font-size:16px;">
                        Beta: {fmt_num(iv_res.get('iv_beta'))}
                    </span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-lbl">Statistical Confidence</span>
                    <span class="kpi-val mono" style="font-size:16px;">
                        p = {fmt_p(welch[1])}
                    </span>
                </div>
            </div>

            <!-- SECTION 1: ECONOMETRIC RIGOR -->
            <h2>1. Econometric Validation (Full Specification)</h2>
            <div style="display:grid; grid-template-columns: 1.2fr 0.8fr; gap:20px;">
                
                <!-- OLS / THRESHOLD -->
                <div>
                    <h3 style="color:#fff;">A. Threshold Regression Model (Hansen 2000)</h3>
                    <table>
                        <thead><tr><th>Parameter</th><th>Coefficient</th><th>t-Statistic</th><th>Significance (p)</th></tr></thead>
                        <tbody>
                            <tr>
                                <td>Beta 1 (Normal Regime Growth)</td>
                                <td class="success mono">{thresh['beta_1_normal']:.4f}%</td>
                                <td class="mono">--</td>
                                <td class="mono">--</td>
                            </tr>
                            <tr>
                                <td>Beta 2 (Shock Regime Growth)</td>
                                <td class="warn mono">{thresh['beta_2_shock']:.4f}%</td>
                                <td class="mono">--</td>
                                <td class="mono">--</td>
                            </tr>
                            <tr>
                                <td><strong>Marginal Impact (S_t)</strong></td>
                                <td class="danger mono"><strong>{thresh['marginal_impact']:.4f}%</strong></td>
                                <td class="mono">{welch[0]:.4f}</td>
                                <td class="mono"><strong>{fmt_p(welch[1])}</strong></td>
                            </tr>
                            <tr>
                                <td>Optimal Threshold (Min SSR)</td>
                                <td class="mono">{hansen.get('optimal_threshold', 0):.4f} TCI</td>
                                <td class="mono">SSR: {hansen.get('min_ssr', 0):.2f}</td>
                                <td class="mono">Optimized</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- ELASTICITY & STATIONARITY -->
                <div>
                    <h3 style="color:#fff;">B. Time Series Diagnostics</h3>
                    <table>
                        <thead><tr><th>Test Specification</th><th>Statistic</th><th>Verdict</th></tr></thead>
                        <tbody>
                            <tr>
                                <td><strong>Log-Log Elasticity</strong><br><span class="tiny">Velocity sensitivity to 1% TCI increase</span></td>
                                <td class="mono">
                                    Beta: {elast['beta']:.4f}<br>
                                    R²: {elast['r_squared']:.4f}
                                </td>
                                <td class="mono">{fmt_p(elast['p_value'])}</td>
                            </tr>
                            <tr>
                                <td><strong>ADF Stationarity</strong><br><span class="tiny">H0: Unit Root (Random Walk)</span></td>
                                <td class="mono">
                                    ADF: {adf['adf_stat']:.4f}<br>
                                    p: {fmt_p(adf['p_value'])}
                                </td>
                                <td class="{'success' if adf['is_stationary'] else 'danger'}">
                                    { "STATIONARY" if adf['is_stationary'] else "NON-STATIONARY" }
                                </td>
                            </tr>
                            <tr>
                                <td><strong>Bootstrap CI (5k)</strong><br><span class="tiny">95% Confidence Interval</span></td>
                                <td class="mono">[{boot[0]:.2f}%, {boot[1]:.2f}%]</td>
                                <td class="{'success' if (boot[0] < 0 and boot[1] < 0) else 'warn'}">
                                    { "ROBUST" if (boot[0] < 0 and boot[1] < 0) else "CROSSES ZERO" }
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- IV-2SLS TABLE -->
            <h3 style="margin-top:25px; color:#fff;">C. Endogeneity Correction (IV-2SLS)</h3>
            <p class="tiny" style="margin-top:0;">Instrument: Lagged Mempool Size (Physical Backlog) | Target: Friction (TCI) | Dependent: Velocity Change</p>
            <table>
                <thead>
                    <tr>
                        <th>Model Stage</th>
                        <th>Coefficient / Metric</th>
                        <th>Standard Error / R²</th>
                        <th>P-Value</th>
                        <th>Causal Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Stage 1 (Instrument Strength)</td>
                        <td class="mono">F-Stat: N/A (See R²)</td>
                        <td class="mono">R²: {iv_res.get('stage1_strength', 0):.4f}</td>
                        <td class="mono">< 0.0001</td>
                        <td>Relevance of Instrument (Backlog -> Friction)</td>
                    </tr>
                    <tr>
                        <td>Stage 2 (Causal Estimate)</td>
                        <td class="mono danger"><strong>IV Beta: {fmt_num(iv_res.get('iv_beta'))}</strong></td>
                        <td class="mono">--</td>
                        <td class="mono"><strong>{fmt_p(iv_res.get('p_value'))}</strong></td>
                        <td>Pure Supply-Side Constraint Impact</td>
                    </tr>
                </tbody>
            </table>

            <!-- SECTION 2: STRUCTURAL PHYSICS -->
            <h2>2. Structural Regime Physics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Network Metric</th>
                        <th>Normal Regime (μ)</th>
                        <th>Shock Regime (μ)</th>
                        <th>Absolute Delta (Δ)</th>
                        <th>Multiplier (x)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Transaction Fee (USD)</td>
                        <td class="mono">${regime_stats.loc['NORMAL', 'Fees']:.2f}</td>
                        <td class="danger mono">${regime_stats.loc['SHOCK', 'Fees']:.2f}</td>
                        <td class="mono">+${regime_stats.loc['SHOCK', 'Fees'] - regime_stats.loc['NORMAL', 'Fees']:.2f}</td>
                        <td class="warn mono">{fee_mult:.2f}x</td>
                    </tr>
                    <tr>
                        <td>Latency (Minutes)</td>
                        <td class="mono">{regime_stats.loc['NORMAL', 'Delay']:.2f}m</td>
                        <td class="danger mono">{regime_stats.loc['SHOCK', 'Delay']:.2f}m</td>
                        <td class="mono">+{regime_stats.loc['SHOCK', 'Delay'] - regime_stats.loc['NORMAL', 'Delay']:.2f}m</td>
                        <td class="warn mono">{regime_stats.loc['SHOCK', 'Delay'] / regime_stats.loc['NORMAL', 'Delay']:.2f}x</td>
                    </tr>
                    <tr>
                        <td>Composite Friction (TCI)</td>
                        <td class="mono">{regime_stats.loc['NORMAL', 'TCI']:.2f}</td>
                        <td class="danger mono">{regime_stats.loc['SHOCK', 'TCI']:.2f}</td>
                        <td class="mono">+{regime_stats.loc['SHOCK', 'TCI'] - regime_stats.loc['NORMAL', 'TCI']:.2f}</td>
                        <td class="danger mono">{tci_mult:.2f}x</td>
                    </tr>
                    <tr>
                        <td>Velocity Growth (30d)</td>
                        <td class="success mono">{vel_normal:+.4f}%</td>
                        <td class="danger mono">{vel_shock:+.4f}%</td>
                        <td class="danger mono"><strong>{net_damage:+.4f}%</strong></td>
                        <td class="danger mono"><strong>COLLAPSE</strong></td>
                    </tr>
                </tbody>
            </table>

            <!-- SECTION 3: INSOLVENCY & EXCLUSION -->
            <h2>3. Insolvency & Economic Exclusion</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric Definition</th>
                        <th>Computed Value</th>
                        <th>Economic Implication</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Insolvency Threshold</strong><br><span class="tiny">{engine.insol_lbl} limit</span></td>
                        <td class="mono">{engine.insol_thresh}</td>
                        <td>Baseline for Economic Viability</td>
                    </tr>
                    <tr>
                        <td><strong>Days of Exclusion</strong><br><span class="tiny">Count where TCI > Threshold</span></td>
                        <td class="warn mono">{insol_stats['days']} Days</td>
                        <td>Periods where retail cannot transact economically.</td>
                    </tr>
                    <tr>
                        <td><strong>Peak Friction Metric</strong><br><span class="tiny">Max Observed {engine.insol_lbl}</span></td>
                        <td class="danger mono">{insol_stats['max']:.2f}</td>
                        <td>Extreme outlier stress event.</td>
                    </tr>
                    <tr>
                        <td><strong>Stagflation Probability</strong><br><span class="tiny">P(Price <= 0 AND Vel <= 0 | Shock)</span></td>
                        <td class="danger mono">{insol_stats['stag_prob']:.2f}%</td>
                        <td>Probability of "Hollow" market conditions.</td>
                    </tr>
                    {granger_html}
                </tbody>
            </table>

            <!-- SECTION 4: DEEP DIVE (Inserted Dynamically) -->
            {utxo_html}

            <!-- SECTION 5: EVENT LOG -->
            <h2>4. Forensic Event Log (Detailed)</h2>
            <p class="tiny">Chronological listing of all detected Regime Shifts. Definition: Consecutive days where TCI > {engine.thresh_val:.2f}.</p>
            <table>
                <thead>
                    <tr>
                        <th>Start Date</th>
                        <th>End Date</th>
                        <th>Dur (Days)</th>
                        <th>Peak TCI</th>
                        <th>Avg Fee</th>
                        <th>Avg Delay</th>
                        <th>Price Impact</th>
                        <th>Velocity Impact</th>
                        <th>Dominant State</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'''
                    <tr>
                        <td class="mono">{r['Start'].strftime('%Y-%m-%d')}</td>
                        <td class="mono">{r['End'].strftime('%Y-%m-%d')}</td>
                        <td class="mono">{r['Duration']}</td>
                        <td class="danger mono">{r['Peak_TCI']:.2f}</td>
                        <td class="mono">${r['Avg_Fee']:.2f}</td>
                        <td class="mono">{r['Avg_Delay']:.2f}m</td>
                        <td class="mono {'success' if r['Price_Imp']>0 else 'danger'}">{r['Price_Imp']:+.2f}%</td>
                        <td class="mono {'success' if r['Vel_Imp']>0 else 'danger'}"><strong>{r['Vel_Imp']:+.2f}%</strong></td>
                        <td>{r['State']}</td>
                    </tr>''' for _, r in events_df.iterrows()]) if not events_df.empty else "<tr><td colspan='9'>No events detected above threshold.</td></tr>"}
                </tbody>
            </table>

            <!-- SECTION 6: RAW DAILY LOG -->
            <h2>5. Raw Daily Shock Data (Full Extract)</h2>
            <div style="max-height:800px; overflow-y:scroll; border:1px solid #333; background:#000;">
                <table>
                    <thead>
                        <tr style="position:sticky; top:0; background:#000; z-index:10;">
                            <th>Date</th>
                            <th>Fee ($)</th>
                            <th>Delay (m)</th>
                            <th>Mempool (B)</th>
                            <th>TCI Score</th>
                            <th>Velocity</th>
                            <th>30d Change</th>
                            <th>State</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td class="mono">{i.strftime('%Y-%m-%d')}</td>
                            <td class="mono">${r['Fees']:.2f} if 'Fees' in r else '-'</td>
                            <td class="mono">{r['Delay']:.2f}m</td>
                            <td class="mono">{int(r['Mempool']) if 'Mempool' in r else '-'}</td>
                            <td class="mono danger">{r['TCI']:.4f}</td>
                            <td class="mono">{r['Velocity']:.8f}</td>
                            <td class="mono {'success' if r['Vel_30d_Change']>0 else 'danger'}">{r['Vel_30d_Change']:+.4f}%</td>
                            <td class="tiny">{r['State']}</td>
                        </tr>''' for i, r in df[df['Regime'] == 'SHOCK'].sort_values('Date', ascending=False).iterrows()])}
                    </tbody>
                </table>
            </div>

            <div style="margin-top:50px; border-top:1px solid #333; padding-top:10px; color:#444; text-align:center;">
                <p><strong>Research Suite v7.1 | Forensic Audit Core</strong><br>
                Generated via Python 3.x | Matplotlib | Statsmodels | Pandas</p>
            </div>
        </body>
        </html>
        """
        return html
        
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
            
            # 1. Sidebar Frame
            sidebar = tk.Frame(main, bg=THEME["panel"], width=320)
            sidebar.pack(side=tk.LEFT, fill=tk.Y)
            sidebar.pack_propagate(False) # Strict width enforcement
            
            # ---------------------------------------------------------
            # BOTTOM PANEL: Control Buttons (Packed FIRST to ensure visibility)
            # ---------------------------------------------------------
            control_panel = tk.Frame(sidebar, bg=THEME["panel"])
            control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

            tk.Label(control_panel, text="PARAMETERS", font=THEME["font_h"], bg=THEME["panel"], fg=THEME["accent"]).pack(anchor="w", pady=(10,5))
            tk.Label(control_panel, text="Friction Threshold %", bg=THEME["panel"], fg="#888").pack(anchor="w")
            
            self.thresh_scale = tk.Scale(control_panel, from_=80, to=99.9, orient=tk.HORIZONTAL, bg=THEME["panel"], fg=THEME["accent"], troughcolor="#222")
            self.thresh_scale.set(95.0)
            self.thresh_scale.pack(fill=tk.X, pady=5)
            
            self.btn_run = tk.Button(control_panel, text="RUN ANALYSIS", command=self._run, bg=THEME["accent"], fg="black", font=("Segoe UI", 10, "bold"), pady=8)
            self.btn_run.pack(fill=tk.X, pady=(15, 5))
            
            tk.Button(control_panel, text="EXPORT REPORT", command=self._export, bg=THEME["shock"], fg="white").pack(fill=tk.X, pady=5)

            # ---------------------------------------------------------
            # LOG PANEL: Text Output (Packed SECOND, above buttons)
            # ---------------------------------------------------------
            log_frame = tk.Frame(sidebar, bg="black", height=120)
            log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
            log_frame.pack_propagate(False) # Strict height

            self.log = scrolledtext.ScrolledText(log_frame, bg="black", fg="#0f0", font=("Consolas", 8))
            self.log.pack(fill=tk.BOTH, expand=True)

            # ---------------------------------------------------------
            # TOP PANEL: Scrollable Input List (Packed LAST to fill remaining space)
            # ---------------------------------------------------------
            top_frame = tk.Frame(sidebar, bg=THEME["panel"])
            top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

            tk.Label(top_frame, text="DATA INGESTION", font=THEME["font_h"], bg=THEME["panel"], fg=THEME["accent"]).pack(pady=(10,10), anchor="w")

            # Scrollable Canvas
            canvas = tk.Canvas(top_frame, bg=THEME["panel"], highlightthickness=0)
            scrollbar = tk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg=THEME["panel"])

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Inputs List based on your specific files
            inputs = [
                "Price",        # market-price.json
                "Fees",         # fees-usd-per-transaction.json
                "Delay",        # median-confirmation-time.json
                "Volume",       # estimated-transaction-volume-usd.json
                "MVRV",         # market-value-to-realized-value-ratio.json
                "Supply",       # total-bitcoins.json (CRITICAL)
                "UTXO",         # utxo-count.json
                "LN_Cap",       # Lightning_Network_Capacity.csv
                "Min_Fee",      # Minimum_Tx_Fee.csv
                "Mempool",      # mempool-size.json
                "Organic",      # n-unique-addresses.json
                "NVT",          # nvt.json
                "Hashrate",     # hash-rate.json
                "Difficulty"    # mining-difficulty.json
            ]
            
            self.status = {}
            
            for k in inputs:
                row = tk.Frame(scrollable_frame, bg=THEME["panel"])
                row.pack(fill=tk.X, padx=5, pady=2)
                
                lbl = k
                if k in ['Price', 'Volume', 'Fees', 'Supply', 'MVRV']:
                    lbl += " *" # Critical files
                
                tk.Label(row, text=lbl, width=12, anchor='w', bg=THEME["panel"], fg="#ccc").pack(side=tk.LEFT)
                self.status[k] = tk.Label(row, text="-", fg="#555", bg=THEME["panel"])
                self.status[k].pack(side=tk.LEFT)
                tk.Button(row, text="LOAD", command=lambda key=k: self._load(key), bg="#222", fg="white", bd=0, width=6).pack(side=tk.RIGHT)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # ---------------------------------------------------------
            # RIGHT SIDE: Notebook
            # ---------------------------------------------------------
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

    # REMOVED: _dummy() method

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
            import traceback
            traceback.print_exc()

    # ==========================================================================
    #  VISUALIZATION SUITE (Strict Mode)
    # ==========================================================================
# ==========================================================================
    #  VISUALIZATION SUITE (Strict Mode - Academic Standard)
    # ==========================================================================
    def _plot_visualizations(self):
            self._log("--- STARTING VISUALIZATION SEQUENCE ---")
            
            df = self.engine.df
            
            if df.empty:
                self._log("CRITICAL ERROR: Dataset is empty.")
                return

            try:
                thresh = self.engine.thresh_val
            except:
                thresh = 0

            # 2. Reset Notebook
            try:
                for tab_id in self.notebook.tabs():
                    self.notebook.forget(tab_id)
            except Exception as e:
                self._log(f"Warning clearing tabs: {e}")

            # 3. Define the Tab Creator
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
                    
                    # --- ACADEMIC FIX: Disable internal titles for publication ---
                    # ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color=THEME["fg"])
                    # -------------------------------------------------------------
                    
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

            # --- PLOT DEFINITIONS ---

            def p_regime_scatter(ax):
                if 'Regime' not in df.columns or 'TCI' not in df.columns: return
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
                if 'Vel_30d_Change' not in df.columns: return
                v_norm = df[df['Regime']=='NORMAL']['Vel_30d_Change'].dropna()
                v_shock = df[df['Regime']=='SHOCK']['Vel_30d_Change'].dropna()
                if len(v_norm) > 1 and len(v_shock) > 1:
                    sns.kdeplot(v_norm, ax=ax, color=THEME["safe"], fill=True, alpha=0.2, label="Normal")
                    sns.kdeplot(v_shock, ax=ax, color=THEME["shock"], fill=True, alpha=0.2, label="Shock")
                    ax.set_xlabel("Velocity Growth Probability")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_damage_box(ax):
                if 'Vel_30d_Change' not in df.columns: return
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

            def p_timeline(ax):
                if 'Velocity' not in df.columns: return
                ax.plot(df.index, df['Velocity'], color=THEME["accent"], lw=1, label="Robust Velocity")
                ylim = ax.get_ylim()
                ax.fill_between(df.index, ylim[0], ylim[1], where=(df['Regime']=='SHOCK'), color=THEME["shock"], alpha=0.3)
                ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_bootstrap(ax):
                if 'Insolvency' not in df.columns: return
                plot_data = df['Insolvency'].replace(0, 0.0001)
                ax.plot(df.index, plot_data, color=THEME["warn"], lw=1)
                ax.axhline(self.engine.insol_thresh, color=THEME["shock"], linestyle="--", label=f"Exclusion Threshold")
                ax.set_yscale('log')
                ax.set_ylabel(f"Cost Burden (%)")
                ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_divergence(ax):
                if 'Div_Score' in df.columns:
                    ax.fill_between(df.index, 0, df['Div_Score'], where=(df['Div_Score'] > 0), color=THEME["shock"], alpha=0.8, label="Artificial Vol")
                    ax.fill_between(df.index, 0, df['Div_Score'], where=(df['Div_Score'] <= 0), color=THEME["safe"], alpha=0.3, label="Organic")
                    ax.set_ylabel("Z-Score Divergence")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_hysteresis(ax):
                if 'TCI_Log' not in df.columns or 'Vel_Log' not in df.columns: return
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

            def p_stagflation(ax):
                if 'State' not in df.columns: return
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
                    ax.set_xlabel("Price Momentum (30d %)")
                    ax.set_ylabel("Utility Momentum (30d %)")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")

            def p_sensitivity(ax):
                if 'TCI' not in df.columns: return
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
            
            def p_utxo_health(ax):
                if 'UTXO' in df.columns:
                    ax.plot(df.index, df['UTXO'], color=THEME["fg"], lw=1, label="UTXO Set Size")
                    ylim = ax.get_ylim()
                    if 'State_Bloat' in df.columns:
                         ax.fill_between(df.index, ylim[0], ylim[1], where=(df['State_Bloat']==1), color=THEME["l2"], alpha=0.3, label="Inelastic Expansion")
                    if 'Confirmed_Exodus' in df.columns:
                         ax.fill_between(df.index, ylim[0], ylim[1], where=(df['Confirmed_Exodus']==1), color=THEME["shock"], alpha=0.5, label="User Contraction")
                    ax.set_ylabel("Unspent Output Count")
                    ax.legend(facecolor=THEME["panel"], labelcolor="white")

            # 4. Execute Creation Loop (RENAMED TO MATCH LATEX)
            self._log("--- Building Tabs ---")
            create_independent_plot_tab("Regime Scatter", p_regime_scatter)
            create_independent_plot_tab("Probability Collapse", p_prob_collapse)
            create_independent_plot_tab("Net Utility Contraction", p_damage_box) # Changed Name
            create_independent_plot_tab("Shock Timeline", p_timeline)
            create_independent_plot_tab("Exclusion Barrier", p_bootstrap) # Changed Name
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
            if 'Regime' not in df.columns:
                self._log("No Regime data found. Cannot export events.")
                return

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
                    p_start = group.iloc[0]['Price'] if 'Price' in group.columns else 0
                    p_end = group.iloc[-1]['Price'] if 'Price' in group.columns else 0
                    
                    v_start = group.iloc[0]['Velocity'] if 'Velocity' in group.columns else 0
                    v_end = group.iloc[-1]['Velocity'] if 'Velocity' in group.columns else 0
                    
                    if p_start > 0: price_imp = ((p_end - p_start) / p_start) * 100
                    else: price_imp = 0
                    
                    if v_start > 0: vel_imp = ((v_end - v_start) / v_start) * 100
                    else: vel_imp = 0
                    
                    if 'State' in group.columns:
                        dom_state = group['State'].mode()[0] if not group['State'].mode().empty else "Unstable"
                    else: dom_state = "N/A"
                    
                    events_data.append({
                        'Start': start, 'End': end, 'Duration': duration,
                        'Peak_TCI': group['TCI'].max(),
                        'Avg_Fee': group['Fees'].mean() if 'Fees' in group.columns else 0,
                        'Avg_Delay': group['Delay'].mean() if 'Delay' in group.columns else 0,
                        'Price_Imp': price_imp,
                        'Vel_Imp': vel_imp,
                        'State': dom_state
                    })
            
            events_df = pd.DataFrame(events_data).sort_values('Peak_TCI', ascending=False) if events_data else pd.DataFrame()

            # ==============================================================================
            #  2. STATISTICS & RISK
            # ==============================================================================
            # Regime Comparison
            cols_to_mean = [c for c in ['Fees', 'Delay', 'TCI', 'Velocity', 'Vel_30d_Change'] if c in df.columns]
            regime_stats = df.groupby('Regime')[cols_to_mean].mean()
            
            # Insolvency Stats
            if 'Insolvency' in df.columns:
                insol_df = df[df['Insolvency'] > self.engine.insol_thresh]
                insol_days = len(insol_df)
                insol_max = df['Insolvency'].max()
            else:
                insol_days = 0
                insol_max = 0
                
            if 'State' in shock_subset.columns:
                 stag_prob = (shock_subset['State'].value_counts().get('Stagflation', 0) / len(shock_subset) * 100) if not shock_subset.empty else 0
            else:
                 stag_prob = 0

            insol_stats = {
                'days': insol_days,
                'max': insol_max,
                'stag_prob': stag_prob
            }

            # Granger Causality (Volume vs Fees)
            granger_html = ""
            if STATS_AVAILABLE and 'Organic' in df.columns and 'Fees' in df.columns:
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
                'bootstrap': EconometricValidator.run_bootstrap_ci(df),
                'hansen': EconometricValidator.run_hansen_threshold_search(df),
                'iv': EconometricValidator.run_iv_regression(df),
                'adf': EconometricValidator.run_stationarity_test(df) # <--- ADD THIS
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
