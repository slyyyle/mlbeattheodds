#!/usr/bin/env python3
"""
Simple DuckDB Browser for Development
Provides a web interface to explore raw data tables.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import duckdb
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

app = Flask(__name__)

class DuckDBBrowser:
    def __init__(self, db_path="data/warehouse.duckdb"):
        self.db_path = db_path
    
    def get_connection(self):
        return duckdb.connect(self.db_path)
    
    def get_schemas(self):
        """Get all schemas in the database."""
        with self.get_connection() as con:
            result = con.execute("SELECT schema_name FROM duckdb_schemas() ORDER BY schema_name").fetchall()
            return [row[0] for row in result]
    
    def get_tables(self, schema='raw'):
        """Get all tables in a schema."""
        with self.get_connection() as con:
            result = con.execute(f"SELECT table_name FROM duckdb_tables() WHERE schema_name='{schema}' ORDER BY table_name").fetchall()
            return [row[0] for row in result]
    
    def get_table_info(self, table_name, schema='raw'):
        """Get column information for a table."""
        with self.get_connection() as con:
            result = con.execute(f"DESCRIBE {schema}.{table_name}").fetchall()
            return [{'column_name': row[0], 'column_type': row[1]} for row in result]
    
    def get_table_sample(self, table_name, schema='raw', limit=50):
        """Get a sample of data from a table."""
        with self.get_connection() as con:
            df = con.execute(f"SELECT * FROM {schema}.{table_name} LIMIT {limit}").df()
            return df
    
    def execute_query(self, query):
        """Execute a custom query."""
        try:
            with self.get_connection() as con:
                df = con.execute(query).df()
                return df, None
        except Exception as e:
            return None, str(e)

db_browser = DuckDBBrowser()

@app.route('/')
def index():
    """Main page showing database overview."""
    schemas = db_browser.get_schemas()
    tables_by_schema = {}
    
    for schema in schemas:
        tables_by_schema[schema] = db_browser.get_tables(schema)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLB Betting Database Browser</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .schema {{ margin: 20px 0; }}
            .table-list {{ margin-left: 20px; }}
            .table-link {{ display: block; margin: 5px 0; text-decoration: none; color: #0066cc; }}
            .table-link:hover {{ text-decoration: underline; }}
            .header {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .query-section {{ margin: 30px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }}
            textarea {{ width: 100%; height: 100px; }}
            button {{ padding: 10px 20px; background: #0066cc; color: white; border: none; border-radius: 3px; cursor: pointer; }}
            button:hover {{ background: #0052a3; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🗄️ MLB Betting Database Browser</h1>
            <p>Explore your DuckDB warehouse data</p>
        </div>
        
        <h2>📊 Database Schemas & Tables</h2>
        {"".join([
            f'''
            <div class="schema">
                <h3>📁 Schema: {schema}</h3>
                <div class="table-list">
                    {"".join([
                        f'<a href="/table/{schema}/{table}" class="table-link">📋 {table}</a>'
                        for table in tables_by_schema[schema]
                    ])}
                </div>
            </div>
            ''' for schema in schemas
        ])}
        
        <div class="query-section">
            <h3>🔍 Custom Query</h3>
            <form action="/query" method="post">
                <textarea name="query" placeholder="SELECT * FROM raw.ml_odds LIMIT 10;"></textarea><br><br>
                <button type="submit">Execute Query</button>
            </form>
        </div>
        
        <h2>🚀 Quick Links</h2>
        <ul>
            <li><a href="/table/raw/ml_odds">💰 Moneyline Odds</a></li>
            <li><a href="/table/raw/so_props">⚾ Strikeout Props</a></li>
            <li><a href="/table/raw/hb_props">🏏 Hits/Total Bases Props</a></li>
            <li><a href="/table/raw/games">🏟️ Games Data</a></li>
            <li><a href="/table/raw/standings">🏆 Standings</a></li>
        </ul>
    </body>
    </html>
    """

@app.route('/table/<schema>/<table_name>')
def view_table(schema, table_name):
    """View a specific table."""
    try:
        # Get table info
        columns = db_browser.get_table_info(table_name, schema)
        
        # Get sample data
        df = db_browser.get_table_sample(table_name, schema)
        
        # Convert DataFrame to HTML
        df_html = df.to_html(classes='data', table_id='dataTable', escape=False)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{schema}.{table_name} - MLB Database Browser</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                .back-link {{ margin: 20px 0; }}
                .table-info {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .data {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .data th, .data td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .data th {{ background-color: #f2f2f2; }}
                .data tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .column {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📋 {schema}.{table_name}</h1>
                <div class="back-link">
                    <a href="/">← Back to Database Overview</a>
                </div>
            </div>
            
            <div class="table-info">
                <h3>📋 Column Information</h3>
                {"".join([
                    f'<div class="column"><strong>{col["column_name"]}</strong>: {col["column_type"]}</div>'
                    for col in columns
                ])}
            </div>
            
            <h3>📊 Sample Data (First 50 rows)</h3>
            {df_html}
            
            <p><em>Showing first 50 rows. Total rows may be much larger.</em></p>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html><body>
            <h1>Error</h1>
            <p>Could not load table {schema}.{table_name}</p>
            <p>Error: {str(e)}</p>
            <a href="/">← Back to Database Overview</a>
        </body></html>
        """

@app.route('/query', methods=['POST'])
def execute_custom_query():
    """Execute a custom SQL query."""
    query = request.form.get('query', '').strip()
    
    if not query:
        return "<html><body><h1>Error</h1><p>No query provided</p><a href='/'>← Back</a></body></html>"
    
    df, error = db_browser.execute_query(query)
    
    if error:
        return f"""
        <html><body>
            <h1>Query Error</h1>
            <p><strong>Query:</strong> {query}</p>
            <p><strong>Error:</strong> {error}</p>
            <a href="/">← Back to Database Overview</a>
        </body></html>
        """
    
    df_html = df.to_html(classes='data', table_id='queryResult', escape=False)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Query Result - MLB Database Browser</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .query-info {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .data {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .data th, .data td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .data th {{ background-color: #f2f2f2; }}
            .data tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Query Results</h1>
            <div><a href="/">← Back to Database Overview</a></div>
        </div>
        
        <div class="query-info">
            <h3>📝 Executed Query</h3>
            <code>{query}</code>
            <p><strong>Rows returned:</strong> {len(df)}</p>
        </div>
        
        <h3>📊 Results</h3>
        {df_html}
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🗄️ Starting MLB Database Browser...")
    print("📡 Access at: http://localhost:5001")
    print("⚠️  This is for development only!")
    app.run(debug=True, port=5001) 