"""
Plan Editor - Apply ML recommendations directly to plans
Modifies plan_rules table with optimization suggestions
"""

import logging
from typing import Dict, Any, Optional, List
import json
from final_db_utils import db_manager

logger = logging.getLogger("plan_editor")


class PlanEditor:
    """
    Applies ML recommendations to actual commission plans
    """
    
    def __init__(self):
        self.db = db_manager
    
    def apply_optimization(self, org_id: int, client_id: int, 
                          plan_id: int, edit_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a specific optimization to a plan
        
        Args:
            org_id: Organization ID
            client_id: Client ID
            plan_id: Plan to modify
            edit_action: Dict with edit instructions from ML recommendation
            
        Returns:
            Dict with success status and details
        """
        logger.info(f"✏️ Applying optimization to plan {plan_id}")
        logger.info(f"Edit action: {edit_action}")
        
        try:
            # Fetch current plan rules
            plan_rules = self._fetch_plan_rules(org_id, client_id, plan_id)
            
            if not plan_rules:
                return {
                    'success': False,
                    'error': f'Plan {plan_id} not found'
                }
            
            # Parse json_data (contains tiers and commission structure)
            json_data = json.loads(plan_rules['json_data']) if plan_rules['json_data'] else []
            
            # Apply edit based on type
            field = edit_action.get('field')
            operation = edit_action.get('operation')
            
            if field == 'quota':
                json_data = self._adjust_quota(json_data, edit_action)
            
            elif field == 'tier_thresholds':
                json_data = self._adjust_tier_threshold(json_data, edit_action)
            
            elif field == 'commission_percentages':
                json_data = self._adjust_commission_rates(json_data, edit_action)
            
            elif field == 'add_tier':
                json_data = self._add_tier(json_data, edit_action)
            
            else:
                return {
                    'success': False,
                    'error': f'Unknown edit field: {field}'
                }
            
            # Update database
            success = self._update_plan_rules(org_id, client_id, plan_id, json_data)
            
            if success:
                return {
                    'success': True,
                    'plan_id': plan_id,
                    'changes_applied': edit_action,
                    'new_structure': json_data
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to update database'
                }
                
        except Exception as e:
            logger.error(f"❌ Edit failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fetch_plan_rules(self, org_id: int, client_id: int, plan_id: int) -> Optional[Dict]:
        """Fetch plan rules from database"""
        query = """
        SELECT pr.* 
        FROM plan_rules pr
        JOIN plan_master pm ON pr.plan_id = pm.id
        WHERE pm.id = %s 
        AND pm.org_id = %s 
        AND pm.client_id = %s
        LIMIT 1
        """
        
        result = self.db.execute_safe_query(
            query,
            params=(plan_id, org_id, client_id),
            tenant_id=org_id,
            client_id=client_id
        )
        
        if result['success'] and result['data']:
            return result['data'][0]
        return None
    
    def _adjust_quota(self, json_data: List[Dict], edit_action: Dict) -> List[Dict]:
        """
        Adjust quota (tier thresholds) by multiplying
        """
        multiplier = edit_action.get('value', 1.0)
        
        for structure in json_data:
            if 'tiers' in structure:
                for tier in structure['tiers']:
                    if 'from_value' in tier and tier['from_value']:
                        try:
                            tier['from_value'] = str(int(float(tier['from_value']) * multiplier))
                        except:
                            pass
                    
                    if 'to_value' in tier and tier['to_value']:
                        try:
                            tier['to_value'] = str(int(float(tier['to_value']) * multiplier))
                        except:
                            pass
        
        logger.info(f"✅ Adjusted quota by {multiplier}x")
        return json_data
    
    def _adjust_tier_threshold(self, json_data: List[Dict], edit_action: Dict) -> List[Dict]:
        """
        Adjust specific tier threshold
        """
        tier_num = edit_action.get('tier', 1)
        adjustment = edit_action.get('value', 0)  # e.g., -0.15 for 15% reduction
        
        for structure in json_data:
            if 'tiers' in structure and len(structure['tiers']) >= tier_num:
                tier = structure['tiers'][tier_num - 1]
                
                # Adjust threshold
                if 'from_value' in tier and tier['from_value']:
                    try:
                        current = float(tier['from_value'])
                        new_value = current * (1 + adjustment)
                        tier['from_value'] = str(int(new_value))
                    except:
                        pass
        
        logger.info(f"✅ Adjusted tier {tier_num} threshold by {adjustment*100:.0f}%")
        return json_data
    
    def _adjust_commission_rates(self, json_data: List[Dict], edit_action: Dict) -> List[Dict]:
        """
        Adjust commission percentages
        """
        operation = edit_action.get('operation', 'subtract')
        value = edit_action.get('value', 0)
        
        for structure in json_data:
            if 'tiers' in structure:
                for tier in structure['tiers']:
                    if 'commission' in tier:
                        try:
                            current_rate = float(tier['commission'])
                            
                            if operation == 'subtract':
                                new_rate = max(0, current_rate - value)
                            elif operation == 'add':
                                new_rate = current_rate + value
                            elif operation == 'multiply':
                                new_rate = current_rate * value
                            else:
                                new_rate = current_rate
                            
                            tier['commission'] = str(round(new_rate, 2))
                        except:
                            pass
        
        logger.info(f"✅ Adjusted commission rates ({operation} {value})")
        return json_data
    
    def _add_tier(self, json_data: List[Dict], edit_action: Dict) -> List[Dict]:
        """
        Add new tier (e.g., accelerator)
        """
        tier_config = edit_action.get('tier_config', {})
        
        for structure in json_data:
            if 'tiers' in structure:
                # Add new tier
                new_tier = {
                    'from_value': str(tier_config.get('threshold', 100)),
                    'to_value': '',
                    'commission': str(tier_config.get('base_rate', 10) * tier_config.get('multiplier', 1.5))
                }
                structure['tiers'].append(new_tier)
        
        logger.info(f"✅ Added new tier: {tier_config}")
        return json_data
    
    def _update_plan_rules(self, org_id: int, client_id: int, 
                          plan_id: int, json_data: List[Dict]) -> bool:
        """
        Update plan_rules table with modified json_data
        """
        query = """
        UPDATE plan_rules pr
        SET json_data = %s::jsonb,
            updated_at = NOW()
        FROM plan_master pm
        WHERE pr.plan_id = pm.id
        AND pm.id = %s
        AND pm.org_id = %s
        AND pm.client_id = %s
        """
        
        try:
            json_str = json.dumps(json_data)
            
            result = self.db.execute_safe_query(
                query,
                params=(json_str, plan_id, org_id, client_id),
                tenant_id=org_id,
                client_id=client_id
            )
            
            if result['success']:
                logger.info(f"✅ Plan {plan_id} updated successfully")
                return True
            else:
                logger.error(f"❌ Update failed: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Update error: {e}")
            return False


# Global instance
plan_editor = PlanEditor()

logger.info("✅ Plan Editor initialized")