import sqlite3
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BLSDatabase:
    def __init__(self, data_dir: str, db_path: str = "bls_oews.db"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def process_file(self, filename: str, table_name: str):
        """Process a file and load it into the database"""
        logger.info(f"Processing {filename} into table {table_name}")
        file_path = self.data_dir / filename

        try:
            # Read the file
            df = pd.read_csv(
                file_path,
                delimiter="\t",
                dtype=str,  # Read everything as string initially
            )

            # Clean column names
            df.columns = df.columns.str.strip()
            logger.info(f"Columns in {table_name}: {list(df.columns)}")

            # For data table, convert numeric columns
            if table_name == "data":
                if "value" in df.columns:
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                if "year" in df.columns:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")

            # Write to database
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            logger.info(f"Created table {table_name} with {len(df)} rows")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            raise

    def create_indexes(self):
        """Create indexes for better query performance"""
        cursor = self.conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_data_series ON data (series_id)",
            "CREATE INDEX IF NOT EXISTS idx_data_year ON data (year)",
            "CREATE INDEX IF NOT EXISTS idx_series_area ON series (area_code)",
            "CREATE INDEX IF NOT EXISTS idx_series_industry ON series (industry_code)",
            "CREATE INDEX IF NOT EXISTS idx_series_occupation ON series (occupation_code)",
        ]

        for idx in indexes:
            try:
                cursor.execute(idx)
                logger.info(f"Created index: {idx}")
            except sqlite3.OperationalError as e:
                logger.error(f"Error creating index: {str(e)}")

        self.conn.commit()

    def create_database(self):
        """Create the complete database"""
        self.connect()

        try:
            # Process all files
            files_to_tables = {
                "oe.area": "area",
                "oe.areatype": "areatype",
                "oe.datatype": "datatype",
                "oe.footnote": "footnote",
                "oe.industry": "industry",
                "oe.occupation": "occupation",
                "oe.release": "release",
                "oe.seasonal": "seasonal",
                "oe.sector": "sector",
                "oe.series": "series",
                "oe.data.0.Current": "data",  # Using current data
            }

            for file, table in files_to_tables.items():
                self.process_file(file, table)

            # Create indexes
            self.create_indexes()

            logger.info("Database creation completed successfully")

        finally:
            self.close()


def main():
    DATA_DIR = "./bls_data"
    db = BLSDatabase(DATA_DIR)
    db.create_database()


if __name__ == "__main__":
    main()
