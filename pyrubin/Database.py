import sqlalchemy
from sqlalchemy import create_engine, text

class Database:
    def __init__(self, database_file):
        self.database_file = database_file
        self.engine = None
        self.connection = None
        self.connect() 
    
    def connect(self):
        # Create the database engine
        self.engine = create_engine(f'sqlite:///{self.database_file}')
        # Establish a connection
        self.connection = self.engine.connect()
    
    def show_tables(self):
        # Show all tables in the database
        result = self.execute_query(text("SELECT name FROM sqlite_master WHERE type='table';"))
        for row in result:
            print(row)

    def disconnect(self):
        # Close the connection
        self.connection.close()
        self.engine.dispose()
    
    def create_table(self, table_name, columns):
        # Create a new table with the specified columns
        self.connection.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({columns})')
    
    def drop_table(self, table_name):
        # Drop an existing table
        self.connection.execute(f'DROP TABLE IF EXISTS {table_name}')
    
    def insert_data(self, table_name, data):
        # Insert data into a table
        values = ', '.join([f"'{value}'" for value in data.values()])
        columns = ', '.join(data.keys())
        self.connection.execute(f'INSERT INTO {table_name} ({columns}) VALUES ({values})')
    
    def execute_query(self, query):
        # Execute a SQL query and return the result
        result = self.connection.execute(query)
        return result
     

if __name__ == "__main__":
    db = Database('occupation-task.db')
    db.show_tables()



