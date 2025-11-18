"""
ML Request Handler - Processes ML predictions and optimizations
Formats responses for chat interface
"""

import logging
from typing import Dict, Any, Optional
from ml_advanced_predictor import tenant_predictor
from intent_detection import extract_ml_params

logger = logging.getLogger("ml_handler")


def handle_ml_request(user_message: str, org_id: int, client_id: int) -> str:
    """
    Main handler for ALL ML requests
    Routes to appropriate ML function and formats response
    """
    logger.info(f"🤖 [ML-HANDLER] Processing: '{user_message}'")
    
    # Extract parameters
    params = extract_ml_params(user_message)
    
    # If plan name provided, look up plan_id
    plan_id = params.get('plan_id')
    if not plan_id and params.get('plan_name'):
        plan_id = lookup_plan_id_by_name(org_id, client_id, params['plan_name'])
        if plan_id:
            logger.info(f"✅ Found plan '{params['plan_name']}' → ID: {plan_id}")
        else:
            return f"❌ I couldn't find a plan named '{params['plan_name']}'. Please check the plan name and try again."
    
    if params['type'] == 'predict' and params['percentage'] is not None:
        # Commission change prediction
        return handle_prediction_request(
            org_id, client_id, params['percentage'], plan_id
        )
    
    elif params['type'] == 'optimize':
        # Plan optimization
        return handle_optimization_request(
            org_id, client_id, plan_id, params.get('plan_name')
        )
    
    else:
        # Could not determine specific request
        return get_ml_help_message()

def handle_prediction_request(org_id: int, client_id: int, 
                              percentage: float, plan_id: Optional[int] = None) -> str:
    """
    Handle commission change prediction
    """
    try:
        result = tenant_predictor.predict_commission_impact(
            org_id=org_id,
            client_id=client_id,
            percentage_change=percentage,
            plan_id=plan_id,
            num_simulations=1000
        )
        
        if not result['success']:
            return f"❌ I couldn't generate predictions: {result.get('error')}"
        
        return format_prediction_response(result)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return f"Sorry, prediction failed: {str(e)}"


def handle_optimization_request(org_id: int, client_id: int,
                                plan_id: Optional[int] = None,
                                plan_name: Optional[str] = None) -> str:
    """
    Handle plan optimization recommendations
    """
    try:
        result = tenant_predictor.recommend_plan_optimizations(
            org_id=org_id,
            client_id=client_id,
            plan_id=plan_id
        )
        
        if not result['success']:
            return f"❌ I couldn't generate recommendations: {result.get('error')}"
        
        return format_optimization_response(result)
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return f"Sorry, optimization failed: {str(e)}"


def format_prediction_response(result: Dict[str, Any]) -> str:
    """
    Format prediction results as markdown
    """
    pct_change = result['requested_change_pct']
    current = result['current_metrics']
    ml_pred = result['ml_predictions']
    sim = result['simulation_analysis']
    risk = result['risk_metrics']
    cohorts = result['cohort_analysis']
    
    direction = "increase" if pct_change > 0 else "decrease"
    
    response = f"## 📊 AI Commission Impact Analysis: {pct_change:+.1f}%\n\n"
    
    response += f"*Analysis for your organization using {result['model_used']} model*\n"
    response += f"*Model Accuracy: R² = {result['model_accuracy']['r2']:.3f}, Error = {result['model_accuracy']['mape']:.1f}%*\n\n"
    
    # Current state
    response += "### 📈 Current State\n"
    response += f"- **Average Payout**: ${current['avg_commission']:,.2f}\n"
    response += f"- **Total Annual Cost**: ${current['total_commission']:,.2f}\n"
    response += f"- **Average Attainment**: {current['avg_attainment']:.1f}%\n"
    response += f"- **Participants**: {current['num_participants']}\n\n"
    
    # ML Predictions
    response += "### 🤖 AI Predicted Impact\n"
    response += f"- **New Average Payout**: ${ml_pred['avg_commission']:,.2f}\n"
    response += f"- **New Total Annual Cost**: ${ml_pred['total_commission']:,.2f}\n"
    response += f"- **Cost Change**: ${ml_pred['total_commission'] - current['total_commission']:+,.2f}\n\n"
    
    # Risk Analysis (Monte Carlo Simulation)
    response += "### ⚠️ Risk Analysis (1,000 Simulations)\n"
    response += f"- **Expected Cost**: ${sim['total_commission']['mean']:,.2f}\n"
    response += f"- **Best Case** (5th percentile): ${sim['total_commission']['percentile_5']:,.2f}\n"
    response += f"- **Worst Case** (95th percentile): ${sim['total_commission']['percentile_95']:,.2f}\n"
    response += f"- **Cost Uncertainty**: ±${sim['total_commission']['std']:,.2f}\n"
    response += f"- **Risk of 10%+ Overrun**: {risk['probability_exceed_10pct']*100:.1f}%\n\n"
    
    # Cohort Analysis
    if cohorts:
        response += "### 👥 Impact by Performance Level\n"
        for cohort in cohorts:
            response += f"- **{cohort['cohort']}** ({cohort['count']} people): "
            response += f"${cohort['current_avg']:,.0f} → ${cohort['predicted_avg']:,.0f} "
            response += f"({cohort['change_pct']:+.1f}%)\n"
        response += "\n"
    
    # Recommendation
    response += "### 💡 AI Recommendation\n"
    response += result['recommendation']
    
    return response


def format_optimization_response(result: Dict[str, Any]) -> str:
    """
    Format optimization recommendations as markdown
    """
    if not result.get('recommendations'):
        return "No optimization opportunities found. Your plans are performing well! 🎉"
    
    response = f"## 🎯 AI Plan Optimization Report\n\n"
    response += f"**{result['summary']}**\n\n"
    
    # High priority plans
    top_plans = [r for r in result['recommendations'] if r['priority_score'] >= 6][:3]
    
    if top_plans:
        response += "### 🔴 High Priority Optimizations\n\n"
        
        for plan in top_plans:
            response += f"#### {plan['plan_name']} (ID: {plan['plan_id']})\n\n"
            
            # Current performance
            perf = plan['current_performance']
            response += "**Current Performance:**\n"
            response += f"- Attainment: {perf['avg_attainment']:.1f}%\n"
            response += f"- Avg Payout: ${perf['avg_commission']:,.2f}\n"
            response += f"- Total Cost: ${perf['total_paid']:,.2f}\n"
            response += f"- Participants: {perf['num_participants']}\n\n"
            
            # Recommendations with edit actions
            response += "**🔧 Recommended Changes:**\n"
            for idx, rec in enumerate(plan['recommendations'], 1):
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                response += f"\n{idx}. {priority_emoji.get(rec['priority'], '⚪')} **{rec['type'].replace('_', ' ').title()}**\n"
                response += f"   - **Issue**: {rec['issue']}\n"
                response += f"   - **Action**: {rec['recommendation']}\n"
                response += f"   - **Expected Impact**: {rec['expected_impact']}\n"
                
                # Add "Apply" option
                if rec.get('edit_action'):
                    response += f"   - ✏️ *Type `apply change {idx} to plan {plan['plan_id']}` to implement this*\n"
            
            response += "\n---\n\n"
    
    # Other plans
    other_plans = [r for r in result['recommendations'] if r['priority_score'] < 6]
    if other_plans:
        response += f"### 🟡 Additional Plans ({len(other_plans)} plans)\n\n"
        response += "Other plans have minor optimization opportunities. "
        response += "Ask me about specific plans for detailed recommendations.\n"
    
    return response


def get_ml_help_message() -> str:
    """
    Help message for ML features
    """
    return """## 🤖 AI Prediction & Optimization Features

I can help you with:

### 📊 Commission Impact Predictions
Forecast what happens when you change commission rates:
- *"Predict the impact of a 10% commission increase"*
- *"What if I decrease commission by 5%?"*
- *"Calculate new payout if I increase plan 123 by 15%"*

### 🎯 Plan Optimization Recommendations
Get AI-powered suggestions to improve your plans:
- *"Recommend optimizations for my plans"*
- *"Analyze plan 456 and suggest improvements"*
- *"How can I improve plan performance?"*

### ✏️ Apply Recommendations Directly
Once I give recommendations, you can apply them:
- *"Apply change 1 to plan 123"*
- *"Implement the quota adjustment for plan 456"*

**Try asking me one of the examples above!**"""


# ========== ADD NEW FUNCTION HERE ==========
def lookup_plan_id_by_name(org_id: int, client_id: int, plan_name: str) -> Optional[int]:
    """
    Look up plan_id by plan name (case-insensitive partial match)
    
    Args:
        org_id: Organization ID
        client_id: Client ID
        plan_name: Plan name to search for
        
    Returns:
        plan_id if found, None otherwise
    """
    from final_db_utils import db_manager
    
    # Search for plans matching the name (case-insensitive, partial match)
    query = """
    SELECT id, program_name 
    FROM plan_master 
    WHERE org_id = %s 
    AND client_id = %s
    AND status = 1
    AND LOWER(program_name) LIKE LOWER(%s)
    ORDER BY created_date DESC
    LIMIT 5
    """
    
    search_term = f"%{plan_name}%"
    
    result = db_manager.execute_safe_query(
        query,
        params=(org_id, client_id, search_term),
        tenant_id=org_id,
        client_id=client_id
    )
    
    if result['success'] and result['data']:
        matches = result['data']
        
        if len(matches) == 1:
            # Exact match found
            logger.info(f"✅ Found 1 plan matching '{plan_name}': {matches[0]['program_name']}")
            return matches[0]['id']
        
        elif len(matches) > 1:
            # Multiple matches - try exact match first
            exact_matches = [m for m in matches if m['program_name'].lower() == plan_name.lower()]
            if exact_matches:
                logger.info(f"✅ Found exact match: {exact_matches[0]['program_name']}")
                return exact_matches[0]['id']
            
            # No exact match - return first result with warning
            logger.warning(f"⚠️ Multiple plans match '{plan_name}': {[m['program_name'] for m in matches]}")
            logger.warning(f"⚠️ Using first match: {matches[0]['program_name']}")
            return matches[0]['id']
    
    logger.error(f"❌ No plan found matching '{plan_name}'")
    return None
# ========== END NEW FUNCTION ==========


# Export functions
__all__ = [
    'handle_ml_request',
    'handle_prediction_request', 
    'handle_optimization_request',
    'format_prediction_response',
    'format_optimization_response',
    'get_ml_help_message',
    'lookup_plan_id_by_name'  # ← UPDATE THIS TOO!
]
