#!/usr/bin/env python3
"""
Quick launcher for the MLB Database Browser
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.db_browser import app

if __name__ == '__main__':
    print("🗄️ Starting MLB Database Browser...")
    print("📡 Opening at: http://localhost:5001")
    print("⚠️  For development use only!")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        app.run(debug=False, port=5001, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\n👋 Database browser stopped.") 