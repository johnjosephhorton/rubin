from flask import Flask, render_template, request, redirect, url_for
import sqlite3

from JobPost import JobPost

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/jobpost/1')

@app.route('/jobpost/<int:job_index>')
def jobpost(job_index):
    # Connect to SQLite database
    conn = sqlite3.connect('job_posts.db')
    cursor = conn.cursor()

    # Fetch the specific job post
    cursor.execute("SELECT id, prompt, ai_generated, edited FROM job_posts WHERE job_index = ?", (job_index,))
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    # Check if the job post was found
    if row:
        job_post = JobPost(*row)
        return job_post.html_table(current_index = job_index, max_index = 1000)
    else:
        return f"No job post found with ID {job_index}"

if __name__ == '__main__':
    app.run(debug=True)
