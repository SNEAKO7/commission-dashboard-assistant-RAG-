"""
Comprehensive ML System Test
Tests all ML features with tenant isolation
"""

import os
from dotenv import load_dotenv
load_dotenv()

from ml_advanced_predictor import tenant_predictor
from plan_editor import plan_editor


def test_prediction():
    """Test commission change prediction"""
    print("\n" + "="*70)
    print("TEST 1: COMMISSION CHANGE PREDICTION")
    print("="*70)
    
    # YOUR ACTUAL IDs
    org_id = 94
    client_id = 93
    
    print(f"\n🏢 Testing for Organization: {org_id}, Client: {client_id}")
    
    # Test 10% increase
    print("\n📈 Testing 10% commission increase...")
    result = tenant_predictor.predict_commission_impact(
        org_id=org_id,
        client_id=client_id,
        percentage_change=10.0,
        num_simulations=1000
    )
    
    if result['success']:
        print("✅ PREDICTION SUCCESS!")
        print(f"   Current avg: ${result['current_metrics']['avg_commission']:,.2f}")
        print(f"   Predicted avg: ${result['ml_predictions']['avg_commission']:,.2f}")
        print(f"   Cost change: ${result['ml_predictions']['total_commission'] - result['current_metrics']['total_commission']:+,.2f}")
        print(f"   Model used: {result['model_used']}")
        print(f"   Model accuracy: R²={result['model_accuracy']['r2']:.3f}")
        print(f"\n   Risk Analysis:")
        print(f"   - Best case: ${result['simulation_analysis']['total_commission']['percentile_5']:,.0f}")
        print(f"   - Worst case: ${result['simulation_analysis']['total_commission']['percentile_95']:,.0f}")
        print(f"   - Probability of 10%+ overrun: {result['risk_metrics']['probability_exceed_10pct']*100:.1f}%")
    else:
        print(f"❌ PREDICTION FAILED: {result.get('error')}")
    
    # Test 5% decrease
    print("\n📉 Testing 5% commission decrease...")
    result = tenant_predictor.predict_commission_impact(
        org_id=org_id,
        client_id=client_id,
        percentage_change=-5.0,
        num_simulations=1000
    )
    
    if result['success']:
        print("✅ PREDICTION SUCCESS!")
        print(f"   Cost savings: ${result['current_metrics']['total_commission'] - result['ml_predictions']['total_commission']:,.2f}")
    else:
        print(f"❌ PREDICTION FAILED: {result.get('error')}")


def test_optimization():
    """Test plan optimization"""
    print("\n" + "="*70)
    print("TEST 2: PLAN OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    org_id = 94
    client_id = 93
    
    print(f"\n🏢 Testing for Organization: {org_id}, Client: {client_id}")
    
    result = tenant_predictor.recommend_plan_optimizations(
        org_id=org_id,
        client_id=client_id
    )
    
    if result['success']:
        print("✅ OPTIMIZATION SUCCESS!")
        print(f"\n   {result['summary']}")
        print(f"\n   Plans analyzed: {result['num_plans_analyzed']}")
        
        if result['recommendations']:
            print(f"\n   📋 Top Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"\n   - {rec['plan_name']} (ID: {rec['plan_id']})")
                print(f"     Priority Score: {rec['priority_score']}")
                print(f"     Issues: {len(rec['recommendations'])}")
                for r in rec['recommendations'][:2]:
                    print(f"       • {r['issue']}")
                    print(f"         → {r['recommendation']}")
    else:
        print(f"❌ OPTIMIZATION FAILED: {result.get('error')}")


def test_model_training():
    """Test model training"""
    print("\n" + "="*70)
    print("TEST 3: MODEL TRAINING")
    print("="*70)
    
    org_id = 94
    client_id = 93
    
    print(f"\n🏢 Training models for Organization: {org_id}, Client: {client_id}")
    
    result = tenant_predictor.train_tenant_models(org_id, client_id)
    
    if result['success']:
        print("✅ TRAINING SUCCESS!")
        print(f"   Best model: {result['best_model']}")
        print(f"   Training samples: {result['training_samples']}")
        print(f"   Test samples: {result['test_samples']}")
        print(f"\n   Model Performance:")
        for model_name, metrics in result['metrics'].items():
            print(f"   - {model_name}:")
            print(f"     R²: {metrics['r2']:.4f}")
            print(f"     MAE: ${metrics['mae']:.2f}")
            print(f"     MAPE: {metrics['mape']:.2f}%")
    else:
        print(f"❌ TRAINING FAILED: {result.get('error')}")


def test_data_isolation():
    """Test tenant data isolation"""
    print("\n" + "="*70)
    print("TEST 4: TENANT DATA ISOLATION")
    print("="*70)
    
    org_id = 94
    client_id = 93
    
    print(f"\n🔒 Testing data isolation for Organization: {org_id}, Client: {client_id}")
    
    # Fetch data
    df = tenant_predictor.fetch_tenant_commissions(org_id, client_id, months_back=12)
    
    if not df.empty:
        print("✅ DATA FETCHED!")
        print(f"   Records: {len(df)}")
        print(f"   Unique org_ids: {df['org_id'].unique()}")
        print(f"   Unique client_ids: {df['client_id'].unique()}")
        
        # Verify isolation
        if len(df['org_id'].unique()) == 1 and df['org_id'].unique()[0] == org_id:
            print("   ✅ ORG ISOLATION VERIFIED")
        else:
            print("   ❌ ORG ISOLATION BREACH!")
        
        if len(df['client_id'].unique()) == 1 and df['client_id'].unique()[0] == client_id:
            print("   ✅ CLIENT ISOLATION VERIFIED")
        else:
            print("   ❌ CLIENT ISOLATION BREACH!")
    else:
        print("❌ NO DATA FETCHED")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 ML SYSTEM COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        test_data_isolation()
        test_model_training()
        test_prediction()
        test_optimization()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()