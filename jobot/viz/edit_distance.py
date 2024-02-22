import csv
from rich.console import Console
import sqlite3
from JobPost import JobPost

# Initialize a rich console

console = Console()

with open("posts_text.csv") as f:
    reader = csv.reader(f)
    data = list(reader)

conn = sqlite3.connect('job_posts.db')
cursor = conn.cursor()

# Query the database to retrieve all rows
cursor.execute("SELECT id, prompt, ai_generated, edited FROM job_posts")
rows = cursor.fetchall()

for example in rows:
    id, prompt, ai_generated, edited = example
    job_post = JobPost(id, prompt, ai_generated, edited)

    job_post.table(console)
    job_post.show_diffs(console)

    # User input for continuation
    user_input = console.input("[bold yellow]Press Enter to continue or type 'q' to quit:[/bold yellow] ")
    if user_input.lower() == 'q':
        console.print("[bold red]Exiting loop.[/bold red]")
        break
    console.clear()
