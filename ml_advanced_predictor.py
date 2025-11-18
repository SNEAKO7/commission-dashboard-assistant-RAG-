"""
Advanced ML Commission Predictor with Tenant Isolation
- Tenant-specific data filtering (org_id + client_id)
- Multiple prediction algorithms (Time Series, Cohort Analysis, Simulation)
- Automatic plan optimization suggestions with edit capability
- Connected to existing database through final_db_utils.py
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import json
from collections import defaultdict

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import xgboost as xgb
import lightgbm as lgb

# Time series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available - time series forecasting disabled")

# Database utilities
from final_db_utils import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_advanced_predictor")


class TenantMLPredictor:
    """
    Advanced ML predictor with STRICT tenant isolation
    All queries are filtered by org_id AND client_id
    """
    
    def __init__(self):
        self.db = db_manager
        
        # Tenant-specific model cache: {(org_id, client_id): models}
        self.tenant_models = {}
        self.tenant_scalers = {}
        self.tenant_feature_cols = {}
        
        logger.info("✅ Tenant-Isolated ML Predictor initialized")
    
    def _get_tenant_key(self, org_id: int, client_id: int) -> str:
        """Generate unique key for tenant"""
        return f"{org_id}_{client_id}"
    
    # ==================== TENANT-SPECIFIC DATA RETRIEVAL ====================
    

    def fetch_tenant_commissions(self, org_id: int, client_id: int, months_back: int = 12) -> pd.DataFrame:
        """
        Fetch commission data for a specific tenant (org + client)
        with strict tenant isolation
        
        Args:
            org_id: Organization ID
            client_id: Client ID  
            months_back: How many months of history to fetch
            
        Returns:
            DataFrame with commission history for this tenant
        """
        cutoff_date = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')

        # CRITICAL: Ensure org_id and client_id are integers (JWT may pass strings)
        org_id = int(org_id)
        client_id = int(client_id)

        logger.info(f"📊 Fetching commission history for ORG:{org_id}, CLIENT:{client_id}")
        
        # Query using actual schema column names from commission_run_details
        # Query using subquery to avoid ambiguous column errors
        query = """
        SELECT
            crd.run_id,
            crd.run_date,
            crd.status,
            crd.org_id,
            crd.client_id,
            crd.plan_master_id as plan_id,
            pm.program_name,
            pm.valid_from,
            pm.valid_to,
            pr.plan_type,
            pr.category_type,
            pr.tiered,
            crd.partner_details_id as employee_id,
            crd.invoice_amount as sales_amount,
            crd.commission_amount as commission_earned,
            crd.total_commission_percent as attainment_percentage,
            crd.slab as tier_achieved
        FROM (
            SELECT *
            FROM commission_run_details
            WHERE org_id = %s
            AND client_id = %s
            AND run_date >= %s
            AND status = 9 
        ) crd
        JOIN plan_master pm ON crd.plan_master_id = pm.id
        LEFT JOIN plan_rules pr ON pm.id = pr.plan_id
        ORDER BY crd.run_date DESC
        """
        
        # Execute query - still passes tenant params for security logging!
        result = self.db.execute_safe_query(
            query, 
            params=(org_id, client_id, cutoff_date),
            tenant_id=org_id,
            client_id=client_id
        )
        
        if not result['success']:
            logger.error(f"Query failed: {result.get('error')}")
            return pd.DataFrame()
        
        data = result['data']
        
        if not data:
            logger.warning(f"No commission data found for tenant {org_id}_{client_id}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # CRITICAL: Convert ALL numeric/decimal columns to float (PostgreSQL returns Decimal objects)
        for col in df.columns:
            # Try to convert to numeric, if it works, it was a number
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        logger.info(f"✅ Fetched {len(df)} records for {org_id}_{client_id}")
        
        # CRITICAL: Verify tenant isolation
        unique_orgs = df['org_id'].unique()
        unique_clients = df['client_id'].unique()
        
        assert len(unique_orgs) == 1 and unique_orgs[0] == org_id, \
            f"TENANT BREACH: Expected only org {org_id}, got {unique_orgs}"
        
        assert len(unique_clients) == 1 and unique_clients[0] == client_id, \
            f"TENANT BREACH: Expected only client {client_id}, got {unique_clients}"
        
        logger.info(f"✅ Tenant isolation verified for {org_id}_{client_id}")
        
        return df

    def fetch_tenant_plans(self, org_id: int, client_id: int, 
                          plan_id: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch plan-level data ONLY for specific tenant
        """
        # CRITICAL: Ensure org_id and client_id are integers
        org_id = int(org_id)
        client_id = int(client_id)

        query = """
        SELECT
            pm.id as plan_id,
            pm.program_name,
            pm.valid_from,
            pm.valid_to,
            pr.plan_type,
            pr.category_type,
            pr.tiered,
            COUNT(DISTINCT crd.partner_details_id) as num_participants,
            AVG(crd.total_commission_percent) as avg_attainment,
            AVG(crd.commission_amount) as avg_commission,
            SUM(crd.commission_amount) as total_commission_paid,
            SUM(crd.invoice_amount) as total_sales
        FROM (
            SELECT *
            FROM plan_master
            WHERE org_id = %s
            AND client_id = %s
            AND status = 9
        ) pm
        LEFT JOIN plan_rules pr ON pm.id = pr.plan_id
        LEFT JOIN commission_run_details crd ON pm.id = crd.plan_master_id
        """
        
        params = [org_id, client_id]
        
        if plan_id:
            query += " AND pm.id = %s"
            params.append(plan_id)
        
        query += """
        GROUP BY pm.id, pm.program_name, pm.valid_from, pm.valid_to, 
                 pr.plan_type, pr.category_type, pr.tiered
        ORDER BY pm.id DESC
        """
        
        try:
            result = self.db.execute_safe_query(
                query, 
                params=tuple(params),
                tenant_id=org_id,
                client_id=client_id
            )
            
            if not result['success']:
                logger.error(f"Failed to fetch plans: {result.get('error')}")
                return pd.DataFrame()
            
            df = pd.DataFrame(result['data'])
            logger.info(f"✅ Loaded {len(df)} tenant-specific plans")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch tenant plans: {e}")
            return pd.DataFrame()
    
    # ==================== ADVANCED FEATURE ENGINEERING ====================
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ADVANCED features for ML models
        - Time-based patterns (seasonality, trends)
        - Employee cohort analysis
        - Plan performance metrics
        - Statistical features
        """
        logger.info("🔧 Engineering advanced features...")
        
        if df.empty:
            return df

        # Convert dates - handle both with and without milliseconds, remove timezones
        df['run_date'] = pd.to_datetime(df['run_date'], format='mixed', errors='coerce').dt.tz_localize(None)
        df['valid_from'] = pd.to_datetime(df['valid_from'], format='mixed', errors='coerce').dt.tz_localize(None)
        df['valid_to'] = pd.to_datetime(df['valid_to'], format='mixed', errors='coerce').dt.tz_localize(None)
        
        # ===== TIME FEATURES =====
        df['year'] = df['run_date'].dt.year
        df['month'] = df['run_date'].dt.month
        df['quarter'] = df['run_date'].dt.quarter
        df['day_of_week'] = df['run_date'].dt.dayofweek
        df['is_month_end'] = df['run_date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['run_date'].dt.is_quarter_end.astype(int)
        df['days_since_epoch'] = (df['run_date'] - pd.Timestamp('2020-01-01')).dt.days
        
        # Cyclical encoding for month (handles seasonality better)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ===== PLAN DURATION & TIMING =====
        df['plan_duration_days'] = (df['valid_to'] - df['valid_from']).dt.days
        df['days_into_plan'] = (df['run_date'] - df['valid_from']).dt.days
        df['plan_progress_pct'] = (df['days_into_plan'] / df['plan_duration_days']).clip(0, 1)
        
        # ===== COMMISSION METRICS =====
        df['commission_rate'] = df['commission_earned'] / (df['sales_amount'] + 1)
        df['sales_per_day'] = df['sales_amount'] / (df['plan_duration_days'] + 1)
        df['commission_per_day'] = df['commission_earned'] / (df['days_into_plan'] + 1)
        
        # ===== CATEGORICAL ENCODING =====
        df['plan_type_encoded'] = pd.Categorical(df['plan_type']).codes
        df['category_type_encoded'] = pd.Categorical(df['category_type']).codes
        df['tiered_flag'] = df['tiered'].astype(int)
        
        # ===== EMPLOYEE-LEVEL AGGREGATIONS =====
        employee_stats = df.groupby('employee_id').agg({
            'commission_earned': ['mean', 'std', 'min', 'max', 'count'],
            'sales_amount': ['mean', 'std', 'sum'],
            'attainment_percentage': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        employee_stats.columns = [
            'employee_id', 
            'emp_avg_commission', 'emp_std_commission', 'emp_min_commission', 'emp_max_commission', 'emp_commission_count',
            'emp_avg_sales', 'emp_std_sales', 'emp_total_sales',
            'emp_avg_attainment', 'emp_std_attainment', 'emp_min_attainment', 'emp_max_attainment'
        ]
        
        df = df.merge(employee_stats, on='employee_id', how='left')
        
        # ===== PLAN-LEVEL AGGREGATIONS =====
        plan_stats = df.groupby('plan_id').agg({
            'commission_earned': ['mean', 'std', 'count', 'sum'],
            'attainment_percentage': ['mean', 'std'],
            'sales_amount': ['mean', 'sum']
        }).reset_index()
        
        plan_stats.columns = [
            'plan_id',
            'plan_avg_commission', 'plan_std_commission', 'plan_num_payouts', 'plan_total_commission',
            'plan_avg_attainment', 'plan_std_attainment',
            'plan_avg_sales', 'plan_total_sales'
        ]
        
        df = df.merge(plan_stats, on='plan_id', how='left')
        
        # ===== LAG FEATURES (Previous performance) =====
        df = df.sort_values(['employee_id', 'run_date'])
        
        for lag in [1, 2, 3]:
            df[f'prev_{lag}_commission'] = df.groupby('employee_id')['commission_earned'].shift(lag)
            df[f'prev_{lag}_sales'] = df.groupby('employee_id')['sales_amount'].shift(lag)
            df[f'prev_{lag}_attainment'] = df.groupby('employee_id')['attainment_percentage'].shift(lag)
        
        # ===== ROLLING AVERAGES =====
        df['rolling_3m_commission'] = df.groupby('employee_id')['commission_earned'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['rolling_3m_sales'] = df.groupby('employee_id')['sales_amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # ===== TREND FEATURES =====
        # Commission growth rate
        df['commission_growth'] = df.groupby('employee_id')['commission_earned'].pct_change()
        df['sales_growth'] = df.groupby('employee_id')['sales_amount'].pct_change()
        
        # ===== COHORT ANALYSIS =====
        # Employee performance tier (based on historical average)
        try:
            df['employee_performance_tier'] = pd.qcut(
                df['emp_avg_commission'], 
                q=4, 
                labels=['Low', 'Medium', 'High', 'Top'],
                duplicates='drop'
            )
        except ValueError:
            # If can't create 4 bins, use fewer bins or median split
            try:
                df['employee_performance_tier'] = pd.qcut(
                    df['emp_avg_commission'], 
                    q=2, 
                    labels=['Low', 'High'],
                    duplicates='drop'
                )
            except:
                # If still fails, just use median
                median = df['emp_avg_commission'].median()
                df['employee_performance_tier'] = df['emp_avg_commission'].apply(
                    lambda x: 'High' if x >= median else 'Low'
                )
        
        df['performance_tier_encoded'] = pd.Categorical(df['employee_performance_tier']).codes

        # ===== STATISTICAL FEATURES =====
        # Z-score (how far from average)
        df['commission_zscore'] = (df['commission_earned'] - df['plan_avg_commission']) / (df['plan_std_commission'] + 1)
        df['attainment_zscore'] = (df['attainment_percentage'] - df['plan_avg_attainment']) / (df['plan_std_attainment'] + 1)
        
        # Fill NaN values - handle categorical columns separately
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['category']).columns.tolist()
        
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        # Don't fill categorical columns (they're already handled)
        
        logger.info(f"✅ Advanced feature engineering complete. Shape: {df.shape}")
        return df

    
    # ==================== MULTI-ALGORITHM PREDICTION ====================
    
    def train_tenant_models(self, org_id: int, client_id: int, 
                           target_col: str = 'commission_earned') -> Dict[str, Any]:
        """
        Train MULTIPLE ML models for specific tenant
        Returns best-performing model
        """
        tenant_key = self._get_tenant_key(org_id, client_id)
        logger.info(f"🤖 Training models for tenant: {tenant_key}")
        
        # Fetch tenant data
        df = self.fetch_tenant_commissions(org_id, client_id, months_back=24)
        
        if df.empty or len(df) < 100:
            logger.warning(f"Insufficient data for tenant {tenant_key}: {len(df)} records")
            return {
                'success': False,
                'error': f'Need at least 100 records, found {len(df)}',
                'tenant_key': tenant_key
            }
        
        # Engineer features
        df = self.engineer_advanced_features(df)
        
        # Select features (using advanced features)
        feature_cols = [
            # Basic metrics
            'sales_amount', 'attainment_percentage', 'commission_rate',
            'sales_per_day', 'commission_per_day',
            
            # Time features
            'year', 'month', 'quarter', 'month_sin', 'month_cos',
            'is_month_end', 'is_quarter_end', 'days_since_epoch',
            
            # Plan features
            'plan_duration_days', 'days_into_plan', 'plan_progress_pct',
            'plan_type_encoded', 'category_type_encoded', 'tiered_flag',
            
            # Employee features
            'emp_avg_commission', 'emp_std_commission', 'emp_avg_sales', 
            'emp_avg_attainment', 'emp_commission_count', 'performance_tier_encoded',
            
            # Plan performance
            'plan_avg_commission', 'plan_avg_attainment', 'plan_std_commission',
            
            # Lag features
            'prev_1_commission', 'prev_2_commission', 'prev_3_commission',
            'prev_1_sales', 'prev_2_sales', 'prev_3_sales',
            
            # Rolling features
            'rolling_3m_commission', 'rolling_3m_sales',
            
            # Trend features
            'commission_growth', 'sales_growth',
            
            # Statistical features
            'commission_zscore', 'attainment_zscore'
        ]
        
        # Filter to available features
        available_features = [f for f in feature_cols if f in df.columns]
        
        X = df[available_features]
        y = df[target_col]
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            logger.warning(f"Only {len(X)} valid samples after filtering")
            return {
                'success': False,
                'error': f'Only {len(X)} valid samples',
                'tenant_key': tenant_key
            }
        
        # Train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ENSEMBLE of models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        results = {}
        best_model = None
        best_score = float('-inf')
        best_model_name = None
        
        for name, model in models.items():
            try:
                logger.info(f"  Training {name}...")
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
                
                results[name] = {
                    'mae': float(mae),
                    'r2': float(r2),
                    'mape': float(mape)
                }
                
                logger.info(f"    {name}: MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                logger.error(f"    {name} training failed: {e}")
        
        if best_model:
            logger.info(f"✅ Best model for {tenant_key}: {best_model_name} (R²={best_score:.4f})")
            
            # Cache model for this tenant
            self.tenant_models[tenant_key] = {
                'model': best_model,
                'model_name': best_model_name,
                'trained_at': datetime.now(),
                'metrics': results[best_model_name],
                'feature_importance': self._get_feature_importance(best_model, available_features)
            }
            self.tenant_scalers[tenant_key] = scaler
            self.tenant_feature_cols[tenant_key] = available_features
            
            return {
                'success': True,
                'tenant_key': tenant_key,
                'best_model': best_model_name,
                'metrics': results,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        
        return {
            'success': False,
            'error': 'All models failed to train',
            'tenant_key': tenant_key
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, [float(x) for x in importances]))
        except:
            pass
        return {}
    
    # ==================== ADVANCED PREDICTION WITH SIMULATION ====================
    
    def predict_commission_impact(self, org_id: int, client_id: int,
                                  percentage_change: float,
                                  plan_id: Optional[int] = None,
                                  num_simulations: int = 1000) -> Dict[str, Any]:
        """
        ADVANCED: Predict commission impact using:
        1. ML model prediction
        2. Monte Carlo simulation for risk analysis
        3. Cohort-based analysis
        4. Statistical confidence intervals
        """
        tenant_key = self._get_tenant_key(org_id, client_id)
        logger.info(f"📈 Predicting {percentage_change}% commission change for {tenant_key}")
        
        # Ensure models are trained
        if tenant_key not in self.tenant_models:
            logger.info(f"Training models for {tenant_key}...")
            train_result = self.train_tenant_models(org_id, client_id)
            if not train_result['success']:
                return {
                    'success': False,
                    'error': train_result.get('error', 'Model training failed')
                }
        
        # Fetch tenant data
        df = self.fetch_tenant_commissions(org_id, client_id, months_back=12)
        
        if df.empty:
            return {
                'success': False,
                'error': 'No commission data available for this tenant'
            }
        
        # Filter by plan if specified
        if plan_id:
            df = df[df['plan_id'] == plan_id]
            if df.empty:
                return {
                    'success': False,
                    'error': f'No data found for plan {plan_id}'
                }
        
        # Engineer features
        df = self.engineer_advanced_features(df)
        
        # ===== BASELINE METRICS =====
        current_stats = {
            'avg_commission': float(df['commission_earned'].mean()),
            'median_commission': float(df['commission_earned'].median()),
            'std_commission': float(df['commission_earned'].std()),
            'total_commission': float(df['commission_earned'].sum()),
            'avg_attainment': float(df['attainment_percentage'].mean()),
            'num_participants': int(df['employee_id'].nunique()),
            'num_records': len(df)
        }
        
        # ===== ML MODEL PREDICTION =====
        multiplier = 1 + (percentage_change / 100)
        df_changed = df.copy()
        df_changed['commission_earned'] = df_changed['commission_earned'] * multiplier
        
        # Recalculate features with changed values
        #df_changed = self.engineer_advanced_features(df_changed)
        
        # Prepare for prediction
        feature_cols = self.tenant_feature_cols[tenant_key]
        X_changed = df_changed[feature_cols]
        X_changed_scaled = self.tenant_scalers[tenant_key].transform(X_changed)
        
        # Predict
        model = self.tenant_models[tenant_key]['model']
        ml_predictions = model.predict(X_changed_scaled)
        
        ml_stats = {
            'avg_commission': float(ml_predictions.mean()),
            'median_commission': float(np.median(ml_predictions)),
            'std_commission': float(ml_predictions.std()),
            'total_commission': float(ml_predictions.sum())
        }
        
        # ===== MONTE CARLO SIMULATION =====
        # Simulate uncertainty in predictions
        simulated_totals = []
        simulated_avgs = []
        
        for _ in range(num_simulations):
            # Add random noise based on historical std
            noise = np.random.normal(0, current_stats['std_commission'], len(ml_predictions))
            sim_predictions = ml_predictions + noise
            sim_predictions = np.maximum(sim_predictions, 0)  # No negative commissions
            
            simulated_totals.append(sim_predictions.sum())
            simulated_avgs.append(sim_predictions.mean())
        
        simulation_stats = {
            'total_commission': {
                'mean': float(np.mean(simulated_totals)),
                'median': float(np.median(simulated_totals)),
                'std': float(np.std(simulated_totals)),
                'min': float(np.min(simulated_totals)),
                'max': float(np.max(simulated_totals)),
                'percentile_5': float(np.percentile(simulated_totals, 5)),
                'percentile_95': float(np.percentile(simulated_totals, 95))
            },
            'avg_commission': {
                'mean': float(np.mean(simulated_avgs)),
                'median': float(np.median(simulated_avgs)),
                'std': float(np.std(simulated_avgs)),
                'min': float(np.min(simulated_avgs)),
                'max': float(np.max(simulated_avgs)),
                'percentile_5': float(np.percentile(simulated_avgs, 5)),
                'percentile_95': float(np.percentile(simulated_avgs, 95))
            }
        }
        
        # ===== COHORT ANALYSIS =====
        df['predicted_commission'] = ml_predictions
        try:
            df['performance_quartile'] = pd.qcut(
                df['emp_avg_commission'], 
                q=4, 
                labels=['Q1-Low', 'Q2-Medium', 'Q3-High', 'Q4-Top'],
                duplicates='drop'
            )
        except ValueError:
            # If can't create 4 bins, use 2 bins
            try:
                df['performance_quartile'] = pd.qcut(
                    df['emp_avg_commission'], 
                    q=2, 
                    labels=['Q1-Low', 'Q2-High'],
                    duplicates='drop'
                )
            except:
                # If still fails, use median
                median = df['emp_avg_commission'].median()
                df['performance_quartile'] = df['emp_avg_commission'].apply(
                    lambda x: 'Q2-High' if x >= median else 'Q1-Low'
                )

        cohort_analysis = df.groupby('performance_quartile').agg({
            'commission_earned': 'mean',
            'predicted_commission': 'mean',
            'employee_id': 'count'
        }).reset_index()
        
        cohort_analysis.columns = ['cohort', 'current_avg', 'predicted_avg', 'count']
        cohort_analysis['change_amount'] = cohort_analysis['predicted_avg'] - cohort_analysis['current_avg']
        cohort_analysis['change_pct'] = (cohort_analysis['change_amount'] / cohort_analysis['current_avg']) * 100
        
        # ===== RISK ANALYSIS =====
        total_cost_change = simulation_stats['total_commission']['mean'] - current_stats['total_commission']
        
        # Probability of cost exceeding threshold
        threshold_10pct = current_stats['total_commission'] * 1.10
        prob_exceed_10pct = sum(t > threshold_10pct for t in simulated_totals) / num_simulations
        
        risk_metrics = {
            'expected_cost_change': float(total_cost_change),
            'cost_change_std': float(simulation_stats['total_commission']['std']),
            'worst_case_cost': float(simulation_stats['total_commission']['percentile_95']),
            'best_case_cost': float(simulation_stats['total_commission']['percentile_5']),
            'probability_exceed_10pct': float(prob_exceed_10pct),
            'confidence_interval_95': [
                float(simulation_stats['total_commission']['percentile_5']),
                float(simulation_stats['total_commission']['percentile_95'])
            ]
        }
        
        # ===== RECOMMENDATION ENGINE =====
        recommendation = self._generate_advanced_recommendation(
            percentage_change,
            current_stats,
            ml_stats,
            simulation_stats,
            cohort_analysis.to_dict('records'),
            risk_metrics
        )
        
        return {
            'success': True,
            'tenant_key': tenant_key,
            'requested_change_pct': percentage_change,
            'current_metrics': current_stats,
            'ml_predictions': ml_stats,
            'simulation_analysis': simulation_stats,
            'cohort_analysis': cohort_analysis.to_dict('records'),
            'risk_metrics': risk_metrics,
            'recommendation': recommendation,
            'model_used': self.tenant_models[tenant_key]['model_name'],
            'model_accuracy': self.tenant_models[tenant_key]['metrics']
        }
    
    def _generate_advanced_recommendation(self, pct_change: float,
                                         current: Dict, ml_pred: Dict,
                                         simulation: Dict, cohorts: List,
                                         risk: Dict) -> str:
        """Generate sophisticated recommendation based on all analyses"""
        
        rec = []
        
        # Overall assessment
        if abs(pct_change) < 5:
            rec.append(f"✅ **Minor Adjustment ({pct_change:+.1f}%)**: This small change is low-risk.")
        elif abs(pct_change) < 15:
            rec.append(f"⚠️ **Moderate Change ({pct_change:+.1f}%)**: Monitor closely for impact.")
        else:
            rec.append(f"🔴 **Major Change ({pct_change:+.1f}%)**: High risk - consider phased rollout.")
        
        # Cost impact
        cost_change = ml_pred['total_commission'] - current['total_commission']
        rec.append(f"\n**Expected Cost Impact**: ${cost_change:+,.0f} annually")
        
        # Risk assessment
        if risk['probability_exceed_10pct'] > 0.3:
            rec.append(f"⚠️ **High Cost Risk**: {risk['probability_exceed_10pct']*100:.0f}% chance costs exceed budget by 10%+")
        
        # Confidence interval
        ci_low, ci_high = risk['confidence_interval_95']
        rec.append(f"**95% Confidence Range**: ${ci_low:,.0f} to ${ci_high:,.0f}")
        
        # Cohort-specific insights
        if cohorts:
            max_impact_cohort = max(cohorts, key=lambda x: abs(x['change_pct']))
            rec.append(f"\n**Biggest Impact**: {max_impact_cohort['cohort']} performers ({max_impact_cohort['change_pct']:+.1f}%)")
        
        # Actionable recommendations
        rec.append("\n**Recommended Actions:**")
        
        if current['avg_attainment'] < 70:
            rec.append("- Low attainment detected - increase may improve motivation")
        elif current['avg_attainment'] > 130:
            rec.append("- High attainment - consider increasing quotas alongside commission change")
        
        if pct_change > 0:
            rec.append("- Monitor retention and engagement metrics post-change")
            rec.append("- Consider performance-based tiers to manage costs")
        else:
            rec.append("- Communicate rationale clearly to maintain morale")
            rec.append("- Offer alternative incentives to offset reduction")
        
        return "\n".join(rec)
    
    # ==================== PLAN OPTIMIZATION WITH EDIT CAPABILITY ====================
    
    def recommend_plan_optimizations(self, org_id: int, client_id: int,
                                    plan_id: Optional[int] = None) -> Dict[str, Any]:
        """
        ADVANCED: Generate actionable plan optimization recommendations
        Each recommendation includes specific edit parameters
        """
        tenant_key = self._get_tenant_key(org_id, client_id)
        logger.info(f"🎯 Generating optimizations for {tenant_key}")
        
        # Fetch plan performance
        plans_df = self.fetch_tenant_plans(org_id, client_id, plan_id)
        commission_df = self.fetch_tenant_commissions(org_id, client_id, months_back=12)
        
        if plans_df.empty:
            return {
                'success': False,
                'error': 'No plan data available for this tenant'
            }
        
        recommendations = []
        
        for idx, plan in plans_df.iterrows():
            plan_id = plan['plan_id']
            plan_name = plan['program_name']
            
            # Filter commission data for this plan
            plan_data = commission_df[commission_df['plan_id'] == plan_id]
            
            if plan_data.empty:
                continue
            
            # Analyze performance
            avg_attainment = float(plan['avg_attainment']) if plan['avg_attainment'] else 0
            avg_commission = float(plan['avg_commission']) if plan['avg_commission'] else 0
            total_paid = float(plan['total_commission_paid']) if plan['total_commission_paid'] else 0
            total_sales = float(plan['total_sales']) if plan['total_sales'] else 0
            
            plan_recs = []
            
            # ===== RECOMMENDATION 1: Quota Adjustment =====
            if avg_attainment < 70:
                optimal_quota_reduction = 100 - avg_attainment
                plan_recs.append({
                    'type': 'quota_adjustment',
                    'priority': 'high',
                    'issue': f'Low attainment ({avg_attainment:.1f}%)',
                    'recommendation': f'Reduce quota by {optimal_quota_reduction:.0f}%',
                    'expected_impact': 'Improve attainment to 80-90% range',
                    'edit_action': {
                        'field': 'quota',
                        'operation': 'multiply',
                        'value': (100 - optimal_quota_reduction) / 100,
                        'apply_to': 'all_tiers'
                    }
                })
            elif avg_attainment > 120:
                optimal_quota_increase = avg_attainment - 100
                plan_recs.append({
                    'type': 'quota_adjustment',
                    'priority': 'medium',
                    'issue': f'High attainment ({avg_attainment:.1f}%)',
                    'recommendation': f'Increase quota by {optimal_quota_increase:.0f}%',
                    'expected_impact': 'Optimize ROI while maintaining motivation',
                    'edit_action': {
                        'field': 'quota',
                        'operation': 'multiply',
                        'value': (100 + optimal_quota_increase) / 100,
                        'apply_to': 'all_tiers'
                    }
                })
            
            # ===== RECOMMENDATION 2: Tier Structure =====
            if plan['tiered']:
                tier_dist = plan_data['tier_achieved'].value_counts()
                total_participants = len(plan_data)
                
                if not tier_dist.empty:
                    # Too many in lowest tier
                    if tier_dist.get(1, 0) / total_participants > 0.5:
                        plan_recs.append({
                            'type': 'tier_threshold',
                            'priority': 'high',
                            'issue': f'{tier_dist.get(1, 0) / total_participants * 100:.1f}% in lowest tier',
                            'recommendation': 'Lower Tier 1 threshold by 15%',
                            'expected_impact': 'Better tier distribution',
                            'edit_action': {
                                'field': 'tier_thresholds',
                                'operation': 'adjust',
                                'tier': 1,
                                'value': -0.15,
                                'type': 'percentage'
                            }
                        })
                    
                    # Too few in top tier
                    max_tier = int(tier_dist.index.max()) if not tier_dist.empty else 0
                    if max_tier > 0 and tier_dist.get(max_tier, 0) / total_participants < 0.05:
                        plan_recs.append({
                            'type': 'tier_threshold',
                            'priority': 'medium',
                            'issue': f'Only {tier_dist.get(max_tier, 0) / total_participants * 100:.1f}% reaching top tier',
                            'recommendation': 'Lower top tier threshold by 10%',
                            'expected_impact': 'Make top tier more achievable',
                            'edit_action': {
                                'field': 'tier_thresholds',
                                'operation': 'adjust',
                                'tier': max_tier,
                                'value': -0.10,
                                'type': 'percentage'
                            }
                        })
            
            # ===== RECOMMENDATION 3: Commission Rate =====
            commission_rate = (total_paid / total_sales * 100) if total_sales > 0 else 0
            
            if commission_rate > 10:
                optimal_rate_reduction = min(2.0, commission_rate - 8)
                plan_recs.append({
                    'type': 'commission_rate',
                    'priority': 'medium',
                    'issue': f'High commission rate ({commission_rate:.2f}%)',
                    'recommendation': f'Reduce commission rates by {optimal_rate_reduction:.1f}%',
                    'expected_impact': f'Annual savings: ${total_paid * (optimal_rate_reduction/commission_rate):,.0f}',
                    'edit_action': {
                        'field': 'commission_percentages',
                        'operation': 'subtract',
                        'value': optimal_rate_reduction,
                        'apply_to': 'all_tiers'
                    }
                })
            
            # ===== RECOMMENDATION 4: Add Accelerators =====
            # Check if high performers could benefit from accelerators
            high_performers = plan_data[plan_data['attainment_percentage'] > 100]
            if len(high_performers) > 0.2 * len(plan_data):
                plan_recs.append({
                    'type': 'add_accelerator',
                    'priority': 'medium',
                    'issue': f'{len(high_performers)} employees exceeding quota',
                    'recommendation': 'Add 1.5x accelerator above 100% attainment',
                    'expected_impact': 'Motivate top performers, increase revenue',
                    'edit_action': {
                        'field': 'add_tier',
                        'operation': 'insert',
                        'tier_config': {
                            'threshold': 100,
                            'multiplier': 1.5,
                            'type': 'accelerator'
                        }
                    }
                })
            
            # Calculate priority score
            priority_score = sum(
                3 if r['priority'] == 'high' else 2 if r['priority'] == 'medium' else 1
                for r in plan_recs
            )
            
            recommendations.append({
                'plan_id': plan_id,
                'plan_name': plan_name,
                'current_performance': {
                    'avg_attainment': round(avg_attainment, 2),
                    'avg_commission': round(avg_commission, 2),
                    'total_paid': round(total_paid, 2),
                    'num_participants': int(plan['num_participants']),
                    'commission_rate': round(commission_rate, 2)
                },
                'recommendations': plan_recs,
                'priority_score': priority_score,
                'can_apply_directly': True  # Flag that edits can be applied
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            'success': True,
            'tenant_key': tenant_key,
            'num_plans_analyzed': len(recommendations),
            'recommendations': recommendations,
            'summary': self._generate_optimization_summary(recommendations)
        }
    
    def _generate_optimization_summary(self, recommendations: List[Dict]) -> str:
        """Generate executive summary"""
        total_plans = len(recommendations)
        high_priority = sum(1 for r in recommendations if r['priority_score'] >= 6)
        
        summary = f"Analyzed {total_plans} commission plans for your organization. "
        summary += f"{high_priority} plans need immediate optimization. "
        
        # Most common issues
        all_recs = [rec for plan in recommendations for rec in plan['recommendations']]
        issue_types = defaultdict(int)
        for rec in all_recs:
            issue_types[rec['type']] += 1
        
        if issue_types:
            top_issue = max(issue_types.items(), key=lambda x: x[1])
            summary += f"Most common issue: {top_issue[0].replace('_', ' ')} ({top_issue[1]} plans)."
        
        return summary


# ==================== GLOBAL INSTANCE ====================

# Initialize once when module is imported
tenant_predictor = TenantMLPredictor()

logger.info("✅ Tenant-Isolated ML Predictor ready")
