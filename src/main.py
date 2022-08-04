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
from zipfile import ZipFile
import pandas as pd
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')
nlp_ner = spacy.load("src/nlp_model")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data  # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
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

    # running through every page
    text = " "
    for page in doc:
        text = text + str(page.get_text())
        text = text.strip()
        text = text.replace("\n", " ")
        # keep only alphanumerics
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text + " "

    custom_ner(text)


def custom_ner(text):
    doc_ = nlp_ner(str(text))
    dict = {}
    selected_dict = {}
    for ent in doc_.ents:
        print(f'{ent.label_} - {ent.text}')
        dict.update({ent.label_: ent.text})
        if ent.label_ == "Years of Experience" or ent.label_ == "Skills" or ent.label_ == "Location" or ent.label_ == "Designation":
            selected_dict.update({ent.label_: ent.text})
    print("dict:", dict)
    print("selected_dict:", selected_dict)

    # creating data frame with required features
    emp_df = pd.DataFrame(selected_dict, index=[0])
    # print(emp_df)
    emp_df.to_csv("resumes_doc.csv")

    df_emp = pd.read_csv(r"michaelres.csv")

    title = " "
    for i in emp_df.columns:
        title1 = emp_df[i]
        title = title + " " + str(title1[0])
        title = re.sub(r'[^a-zA-Z\s]', '', title)
    title = title.split(",")
    print(title)
    claim = " "
    for i in df_emp.columns:
        claim1 = df_emp[i]
        claim = claim + " " + str(claim1[0])
        claim = re.sub(r'[^a-zA-Z\s]', '', claim)
    claim = claim.split(",")
    print(claim)
    title = model.encode(title)
    claim = model.encode(claim)

    cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1))
    print("Similarity: ", cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1)))



##############################################################################################################

    # df_combined = pd.concat([df_emp, emp_df], axis=0)
    # df_combined.reset_index(inplace=True, drop=True)
    # df_combined = df_combined.drop(columns="Unnamed: 0")
    # # print(df_combined)

    # a_list = []
    # for i in range((df_combined.shape[0])):
    #     cur_row = []
    #     for j in range(df_combined.shape[1]):
    #         cur_row.append(df_combined.iat[i, j])
    #     a_list.append(cur_row)
    #
    # # one = ", ".join(a_list[0])
    # # one = re.sub(r'[^a-zA-Z\s]', '', one)
    # print(a_list)
    # res = [' '.join(ele) for ele in a_list]
    # print(res)
    #
    # title = model.encode(res)
    # # claim = model.encode(res)
    #
    # cosine_similarity([title[0]], title[0:])
    # print("Similarity: ", cosine_similarity([title[0]], title[0:]))
############################################################################################


if __name__ == '__main__':
    app.run(debug=True)
