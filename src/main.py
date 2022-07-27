from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm, file
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from pathlib import Path
import sys, fitz
import spacy
import json
from spacy import displacy
from tqdm import tqdm
from spacy.tokens import DocBin
import spacy_transformers
import re
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/'
app.config['READ_FILE'] = 'static/files/input_file'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data#First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        global input_file
        input_file = secure_filename(file.filename)
        print(input_file)

        getPdfData(input_file)

        return "File has been uploaded."
    return render_template('index.html', form=form)


def getPdfData(input_file):
    fullPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], input_file)
    fname = Path(__file__).parent / fullPath
    doc = fitz.open(fname)
    text = " "
    for page in doc:
        text = text + str(page.get_text())

    text = text.strip()

    text = text.replace("\n", "")

    # keep only alphanumerics
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    print(str(text))

    # with open("myfile.txt", "w") as file1:
    #     # Writing data to a file
    #     file1.writelines(text)

    nlp_ner = spacy.load("src/best_model")
    doc = nlp_ner(str(text))
    for ent in doc.ents:
        print(f'{ent.label_} - {ent.text}')


if __name__ == '__main__':
    app.run(debug=True)
