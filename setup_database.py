#!/usr/bin/env python3
"""
Database Setup Script for Augur Seller Scoring
This script initializes the database with the complete schema
"""

import os
import sys
from sqlalchemy import create_engine, text
import streamlit as st
from dotenv import load_dotenv

def setup_database():
    """Setup the database with complete schema"""
    
    # Load environment variables
    load_dotenv()
    
    # Get database credentials - use st.secrets for deployment compatibility
    try:
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]
        db_host = st.secrets["DB_HOST"]
        db_name = st.secrets["DB_NAME"]
        db_port = st.secrets["DB_PORT"]
    except:
        # Fallback to os.getenv for local development
        db_user = os.getenv("DB_USER", "Augur_App")
        db_password = os.getenv("DB_PASSWORD", "augur_password_2024")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "augur")
        db_port = os.getenv("DB_PORT", "5432")
    
    # Create database engine
    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    print(f"🔗 Connecting to database: {db_host}:{db_port}/{db_name}")
    
    try:
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version}")
        
        # Read and execute schema
        schema_file = "db/complete_schema.sql"
        if not os.path.exists(schema_file):
            print(f"❌ Schema file not found: {schema_file}")
            return False
        
        print(f"📖 Reading schema from: {schema_file}")
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        print("🚀 Executing database schema...")
        with engine.begin() as conn:
            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements):
                if statement:
                    try:
                        conn.execute(text(statement))
                        print(f"  ✅ Statement {i+1}/{len(statements)} executed")
                    except Exception as e:
                        print(f"  ⚠️  Statement {i+1} warning: {e}")
        
        print("✅ Database schema setup completed!")
        
        # Verify tables were created
        print("🔍 Verifying tables...")
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            
            print(f"📊 Created {len(tables)} tables:")
            for table in tables:
                print(f"  • {table}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main function"""
    print("🏠 Augur Seller Scoring - Database Setup")
    print("=" * 50)
    
    if setup_database():
        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the application: ./start.sh")
        print("2. Or manually: docker-compose up -d")
        print("3. Access the app at: http://localhost:8501")
    else:
        print("\n❌ Database setup failed!")
        print("\nTroubleshooting:")
        print("1. Check your .env file configuration")
        print("2. Ensure PostgreSQL is running")
        print("3. Verify database credentials")
        sys.exit(1)

if __name__ == "__main__":
    main()
