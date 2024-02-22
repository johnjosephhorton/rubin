import sqlite3
import csv

# Create a connection to the SQLite database
conn = sqlite3.connect('job_posts.db')
cursor = conn.cursor()

# Create or modify the table in the database
cursor.execute('''
    CREATE TABLE IF NOT EXISTS job_posts (
        job_index INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT,
        prompt TEXT,
        ai_generated TEXT,
        edited TEXT
    )
''')

# Read data from the CSV file
with open("posts_text.csv", newline='') as f:
    reader = csv.reader(f)
    next(reader, None)  # Skip the header row if there is one
    for example in reader:
        id = example[1]
        prompt = example[2]
        ai_generated = example[3]
        edited = example[4]

        # Insert data into the database
        cursor.execute('''
            INSERT INTO job_posts (id, prompt, ai_generated, edited)
            VALUES (?, ?, ?, ?)
        ''', (id, prompt, ai_generated, edited))

# Commit the changes and close the database connection
conn.commit()
conn.close()
