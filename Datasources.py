import pyodbc
import pandas as pd

class ELabJobsDB:
    """Handles SQL Server database connections."""

    def __init__(self, server="hchwusrv2062", database="ELabJobs", username="ELabReader",
                 password="sfg7as*TClkc89", driver="{ODBC Driver 17 for SQL Server}"):
        self.config = {
            "server": server,
            "database": database,
            "username": username,
            "password": password,
            "driver": driver
        }

    def get_connection(self):
        """Creates and returns a new database connection."""
        conn_str = (
            f"DRIVER={self.config['driver']};"
            f"SERVER={self.config['server']};"
            f"DATABASE={self.config['database']};"
            f"UID={self.config['username']};"
            f"PWD={self.config['password']}"
        )
        return pyodbc.connect(conn_str)

    def runQuery(self, prompt: str) -> pd.DataFrame:
        """
        Runs a SQL query against the database.
        Returns a pandas DataFrame with the results.
        """
        print(f"🟢 Executing query:\n{prompt}\n")

        try:
            with self.get_connection() as conn:
                df = pd.read_sql(prompt, conn)
                print(f"✅ Query executed successfully: {len(df)} rows, {len(df.columns)} columns")
                return df

        except Exception as e:
            print(f"❌ SQL Error in runQuery: {e}")
            return pd.DataFrame([{"error": str(e)}])

