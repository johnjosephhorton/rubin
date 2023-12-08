from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import csv
import difflib
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class PostDiff(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post_key = db.Column(db.String, unique=True, nullable=False)
    post_desc = db.Column(db.Text, nullable=False)
    generated_description_ai = db.Column(db.Text, nullable=False)
    added_to_post_desc = db.Column(db.Text, nullable=True)
    removed_from_generation = db.Column(db.Text, nullable=True)

db.create_all()