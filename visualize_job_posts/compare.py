import csv
import difflib

def get_diffs(old_text, new_text):
    d = difflib.Differ()
    diffs = list(d.compare(old_text.split(), new_text.split()))
    added = [word[2:] for word in diffs if word.startswith('+ ')]
    removed = [word[2:] for word in diffs if word.startswith('- ')]
    return ' '.join(added), ' '.join(removed)

from nltk.tokenize import sent_tokenize  # You might need to install nltk for this

def get_diffs(old_text, new_text):
    # Split the text into sentences
    old_sentences = sent_tokenize(old_text)
    new_sentences = sent_tokenize(new_text)

    # Find sentence level diffs
    sentence_diffs = list(difflib.Differ().compare(old_sentences, new_sentences))
    added_sentences = [sent[2:] for sent in sentence_diffs if sent.startswith('+ ')]
    removed_sentences = [sent[2:] for sent in sentence_diffs if sent.startswith('- ')]

    # Find word level diffs within sentences
    added_words = []
    removed_words = []
    for old, new in zip(old_sentences, new_sentences):
        if old != new:
            word_diffs = list(difflib.Differ().compare(old.split(), new.split()))
            added_words.extend([word[2:] for word in word_diffs if word.startswith('+ ')])
            removed_words.extend([word[2:] for word in word_diffs if word.startswith('- ')])

    return ' '.join(added_sentences), ' '.join(removed_sentences), ' '.join(added_words), ' '.join(removed_words)


def diff_dictionary():
    # Dictionary to hold the results
    posts_diffs = {}

    # Read the CSV file
    with open('posts_text.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            post_key = row['post_key']
            post_desc = row['post_desc'].replace("\\n", "\n")
            generated_desc = row['generated_description_ai'].replace("\\n", "\n")

            # Get the differences
            added, removed, added_words, removed_words = get_diffs(generated_desc, post_desc)

            # Store in dictionary
            posts_diffs[post_key] = {
                'key': post_key,
                'post_desc': post_desc,
                'generated_description_ai': generated_desc,
                'added_to_post_desc': added,
                'removed_from_generation': removed, 
                'added_words': added_words, 
                'removed_words': removed_words
            }
    return {index:value for index, (key, value) in enumerate(posts_diffs.items())}

import pickle

with open('diffs.pickle', 'wb') as handle:
    # Pickle the dictionary using the highest protocol available
    pickle.dump(diff_dictionary(), handle, protocol=pickle.HIGHEST_PROTOCOL)

