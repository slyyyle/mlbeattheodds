#!/usr/bin/env python3
"""
MLB Betting Analytics Pipeline Orchestrator

This script orchestrates the entire data pipeline:
1. Data ingestion (games, odds)
2. Database initialization
3. Feature engineering
4. Model backtesting
5. API server startup

Usage:
    python scripts/run_pipeline.py [--full] [--skip-ingestion] [--serve-only]
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description, cwd=None):
    """Run a command and handle errors."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Completed: {description}")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def check_environment():
    """Check if required environment variables and dependencies are available."""
    logger.info("Checking environment...")
    
    # Check for required directories
    required_dirs = ['data', 'data/raw', 'data/features', 'data/results']
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check for Python dependencies
    try:
        import pandas
        import duckdb
        import sklearn
        import fastapi
        logger.info("Core dependencies found")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
        return False
    
    # Check for optional dependencies
    try:
        import sportsbet
        logger.info("Sports-betting package found")
    except ImportError:
        logger.warning("Sports-betting package not found. Some features may be limited.")
    
    # Check for environment file
    if not Path('.env').exists():
        logger.warning("No .env file found. Using default configuration.")
        logger.warning("Copy config/env_template.txt to .env and configure for full functionality.")
    
    return True


def run_data_ingestion():
    """Run data ingestion scripts."""
    logger.info("=" * 50)
    logger.info("STARTING DATA INGESTION")
    logger.info("=" * 50)
    
    # Ingest games and standings
    if not run_command(
        "python src/ingest/ingest_games.py",
        "Ingesting MLB games and standings data"
    ):
        return False
    
    # Ingest odds (may fail if no API key)
    run_command(
        "python src/ingest/ingest_odds.py", 
        "Ingesting odds data (optional)"
    )
    
    return True


def initialize_database():
    """Initialize DuckDB warehouse."""
    logger.info("=" * 50)
    logger.info("INITIALIZING DATABASE")
    logger.info("=" * 50)
    
    return run_command(
        "python src/storage/init_db.py --test",
        "Initializing DuckDB warehouse"
    )


def build_features():
    """Build modeling features."""
    logger.info("=" * 50)
    logger.info("BUILDING FEATURES")
    logger.info("=" * 50)
    
    return run_command(
        "python src/features/build_features.py",
        "Building modeling features"
    )


def run_backtests():
    """Run backtesting."""
    logger.info("=" * 50)
    logger.info("RUNNING BACKTESTS")
    logger.info("=" * 50)
    
    return run_command(
        "python src/backtest/run_backtests.py",
        "Running strategy backtests"
    )


def start_api_server():
    """Start the FastAPI server."""
    logger.info("=" * 50)
    logger.info("STARTING API SERVER")
    logger.info("=" * 50)
    
    logger.info("Starting FastAPI server on http://localhost:8000")
    logger.info("API documentation will be available at http://localhost:8000/docs")
    
    # Start API server
    try:
        subprocess.run([
            "python", "-m", "uvicorn", 
            "src.api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("API server stopped")


def start_frontend():
    """Start the React frontend (if available)."""
    ui_dir = Path("ui")
    if ui_dir.exists() and (ui_dir / "package.json").exists():
        logger.info("=" * 50)
        logger.info("STARTING FRONTEND")
        logger.info("=" * 50)
        
        # Install dependencies if needed
        if not (ui_dir / "node_modules").exists():
            logger.info("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=ui_dir)
        
        logger.info("Starting React frontend on http://localhost:3000")
        try:
            subprocess.run(["npm", "start"], cwd=ui_dir)
        except KeyboardInterrupt:
            logger.info("Frontend stopped")
    else:
        logger.info("Frontend not found. Skipping...")


def setup_risk_management():
    """Initialize risk management and bankroll tracking."""
    logger.info("Setting up risk management system...")
    try:
        from ml.risk_management import BankrollManager
        manager = BankrollManager(initial_bankroll=10000)
        logger.info("Risk management system initialized")
        return True
    except Exception as e:
        logger.error(f"Risk management setup failed: {e}")
        return False


def setup_monitoring():
    """Initialize performance monitoring."""
    logger.info("Setting up performance monitoring...")
    try:
        from monitoring.performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        logger.info("Performance monitoring initialized")
        return True
    except Exception as e:
        logger.error(f"Monitoring setup failed: {e}")
        return False


def setup_notifications():
    """Initialize notification system."""
    logger.info("Setting up notification system...")
    try:
        from alerts.notification_system import NotificationSystem
        notifier = NotificationSystem()
        logger.info("Notification system initialized")
        return True
    except Exception as e:
        logger.error(f"Notification setup failed: {e}")
        return False


def initialize_automated_pipeline():
    """Initialize the automated pipeline."""
    logger.info("Initializing automated pipeline...")
    try:
        from pipeline.automated_pipeline import AutomatedMLBPipeline
        config = {
            'data_update_interval_hours': 6,
            'model_retrain_interval_days': 7,
            'performance_check_interval_hours': 1,
            'initial_bankroll': 10000,
            'auto_betting_enabled': False,
            'notification_enabled': True
        }
        pipeline = AutomatedMLBPipeline(config)
        logger.info("Automated pipeline ready for deployment")
        logger.info("To start automated mode, run: python src/pipeline/automated_pipeline.py")
        return True
    except Exception as e:
        logger.error(f"Automated pipeline initialization failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MLB Betting Analytics Pipeline")
    parser.add_argument("--full", action="store_true", 
                       help="Run full pipeline from scratch")
    parser.add_argument("--skip-ingestion", action="store_true",
                       help="Skip data ingestion steps")
    parser.add_argument("--enhanced", action="store_true",
                       help="Run enhanced pipeline with all new features")
    parser.add_argument("--serve-only", action="store_true",
                       help="Only start the API server")
    parser.add_argument("--with-frontend", action="store_true",
                       help="Also start the React frontend")
    
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 60)
    logger.info("MLB BETTING ANALYTICS PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()}")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        sys.exit(1)
    
    if args.serve_only:
        start_api_server()
        return
    
    success = True
    
    # Run pipeline steps
    if not args.skip_ingestion:
        success = success and run_data_ingestion()
    
    if success:
        success = success and initialize_database()
    
    if success:
        success = success and build_features()
    
    if success:
        success = success and run_backtests()
    
    if success:
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        if args.with_frontend:
            # Start both API and frontend
            import threading
            
            # Start API in background thread
            api_thread = threading.Thread(target=start_api_server)
            api_thread.daemon = True
            api_thread.start()
            
            # Start frontend in main thread
            start_frontend()
        else:
            start_api_server()
    else:
        logger.error("Pipeline failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 