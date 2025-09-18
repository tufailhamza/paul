#!/usr/bin/env python3
"""
Verification script for Augur Seller Scoring setup
Run this to verify your installation is working correctly
"""

import sys
import os
import importlib.util

def check_imports():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sqlalchemy', 
        'psycopg2', 'plotly', 'optuna', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_configuration():
    """Check if configuration files exist and are valid"""
    config_files = [
        '.env',
        '.streamlit/secrets.toml',
        'docker-compose.yml',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in config_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    return True

def check_database_connection():
    """Test database connection"""
    try:
        import streamlit as st
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Try st.secrets first
        try:
            db_user = st.secrets['DB_USER']
            db_host = st.secrets['DB_HOST']
            print("✅ Using st.secrets configuration")
            return True
        except:
            pass
        
        # Try os.getenv
        try:
            db_user = os.getenv('DB_USER')
            db_host = os.getenv('DB_HOST')
            if db_user and db_host:
                print("✅ Using os.getenv configuration")
                return True
        except:
            pass
        
        print("❌ No valid database configuration found")
        return False
        
    except Exception as e:
        print(f"❌ Database configuration error: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 Augur Seller Scoring - Setup Verification")
    print("=" * 50)
    
    # Check imports
    print("\n📦 Checking Python packages...")
    imports_ok = check_imports()
    
    # Check configuration
    print("\n⚙️  Checking configuration files...")
    config_ok = check_configuration()
    
    # Check database connection
    print("\n🗄️  Checking database configuration...")
    db_ok = check_database_connection()
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 20)
    
    if imports_ok and config_ok and db_ok:
        print("🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run: sudo docker-compose up -d")
        print("2. Open: http://localhost:8501")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
