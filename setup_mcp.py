#!/usr/bin/env python3
"""
Setup script for MCP Integration with your Flask application.
Run this after updating the files to ensure everything is configured correctly.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup")

def check_file_exists(filename):
    """Check if required file exists"""
    if os.path.exists(filename):
        logger.info(f"✅ {filename} exists")
        return True
    else:
        logger.error(f"❌ {filename} missing")
        return False

def install_dependencies():
    """Install required Python packages"""
    logger.info("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def check_environment_variables():
    """Check if required environment variables are set"""
    logger.info("🔧 Checking environment variables...")
    
    required_vars = [
        'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
        'ANTHROPIC_API_KEY'  # New requirement for MCP
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please update your .env file with the missing variables")
        return False
    else:
        logger.info("✅ All required environment variables are set")
        return True

def test_database_connection():
    """Test database connection"""
    logger.info("🗄️ Testing database connection...")
    
    try:
        from final_db_utils import DatabaseManager
        db_manager = DatabaseManager()
        schema = db_manager.load_schema()
        logger.info(f"✅ Database connected - {len(schema)} tables found")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_anthropic_api():
    """Test Anthropic API connection"""
    logger.info("🤖 Testing Anthropic API connection...")
    
    try:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("❌ ANTHROPIC_API_KEY not set")
            return False
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test with a simple message
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        logger.info("✅ Anthropic API connection successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Anthropic API test failed: {e}")
        return False

def test_mcp_integration():
    """Test MCP integration"""
    logger.info("🔗 Testing MCP integration...")
    
    try:
        from mcp_client import mcp_client
        
        # Test schema loading
        schema_info = mcp_client.get_database_schema()
        logger.info(f"✅ MCP schema loaded: {len(schema_info.get('tables', {}))} tables")
        
        # Test a simple natural language query
        test_result = mcp_client.natural_language_to_sql(
            "show plan types", 
            "test query"
        )
        
        if test_result.get('sql_query'):
            logger.info("✅ MCP natural language to SQL working")
            return True
        else:
            logger.warning("⚠️ MCP working but query generation needs improvement")
            return True
            
    except Exception as e:
        logger.error(f"❌ MCP integration test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up MCP integration for Flask application...")
    
    # Check required files
    required_files = [
        'mcp_client.py', 'mcp_postgres_server.py', 
        'improved_mcp_client.py', 'final_db_utils.py',
        'sql_security.py', 'requirements.txt'
    ]
    
    missing_files = [f for f in required_files if not check_file_exists(f)]
    if missing_files:
        logger.error(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Environment variables loaded")
    except ImportError:
        logger.error("❌ python-dotenv not installed")
        return False
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    if check_environment_variables():
        tests_passed += 1
    
    if test_database_connection():
        tests_passed += 1
    
    if test_anthropic_api():
        tests_passed += 1
    
    if test_mcp_integration():
        tests_passed += 1
    
    # Summary
    logger.info(f"\n📊 Setup Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 Setup completed successfully!")
        logger.info("🚀 You can now run: python app.py")
        return True
    else:
        logger.warning("⚠️ Setup completed with some issues")
        logger.info("Please resolve the issues above before running the application")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
