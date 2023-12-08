from flask import Flask, render_template, request
import random

app = Flask(__name__)


import pickle

# Open the file for reading
with open('diffs.pickle', 'rb') as handle:
    # Load the dictionary back from the pickle file
    posts_diffs = pickle.load(handle)

min_key = min(list(posts_diffs.keys()))
max_key = max(list(posts_diffs.keys()))

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_key = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'submit':
            selected_key = request.form.get('post_key')
        elif action == 'random':
            # Generate a random key (assuming your keys are integers)
            selected_key = random.randint(min_key, max_key)

    if selected_key is not None:
        data = posts_diffs.get(int(selected_key), {})
    else:
        data = {}
    return render_template('index.html', post_keys=posts_diffs.keys(), data=data, selected_key=selected_key)

if __name__ == '__main__':
    #pass
    app.run(debug=True)
