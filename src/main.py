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
import numpy as np
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')
nlp_ner = spacy.load("src/updated model-best")

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

    # Processing docx file
    if input_file.endswith('.docx'):
        temp = docx2txt.process(fname)
        resume_text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        text = ' '.join(resume_text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        custom_ner(text)

    # running through every page
    # pdf
    elif input_file.endswith('.pdf'):
        doc = fitz.open(fname)
        text = " "
        for page in doc:
            text = text + str(page.get_text())
            text = text.strip()
            text = text.replace("\n", " ")
            # keep only alphanumerics
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = text + " "
        custom_ner(text)

    # Processing Zip File
    elif input_file.endswith('.zip'):
        with ZipFile(fname, 'r') as zip:
        # printing all the contents of the zip file
            zip.printdir()
            file_names = zip.namelist()
            zip.extractall()

            list_docx = []
            list_pdf = []
            for j in file_names:
                if j.endswith('.pdf'):
                    print("j", j)
                    doc2 = fitz.open(j)
                    print("doc2: ", doc2)
                    text = " "
                    for page in doc2:
                        text = text + str(page.get_text())
                        text = text.strip()
                        text = text.replace("\n", " ")
                        # keep only alphanumerics
                        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                        text = text + " "
                        # print(text)
                    list_pdf.append(text)

                if j.endswith('.docx'):
                    temp = docx2txt.process(j)
                    resume_text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
                    text = ' '.join(resume_text)
                    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                    list_docx.append(text)

                else:
                    print("File Not Supported")

            list = list_pdf + list_docx

            customZip_ner(list)
    else:
        print("File Not Supported")


def custom_ner(text):
    cosine_df = pd.DataFrame()
    dict = {}
    selected_dict = {}
    doc_ = nlp_ner(str(text))
    for ent in doc_.ents:
        print(f'{ent.label_} - {ent.text}')
        dict.update({ent.label_: ent.text})
        if ent.label_ == "Years of Experience" or ent.label_ == "Skills" or ent.label_ == "Location" or ent.label_ == "Designation":
            selected_dict.update({ent.label_: ent.text})
    print("dict:", dict)
    print("selected_dict:", selected_dict)

    # creating dataframe for cos_sim

    # creating data frame with required features
    emp_df = pd.DataFrame(selected_dict, index=[0])
    # print(emp_df)
    emp_df.to_csv("resumes_doc.csv")

    job_desc = pd.read_csv(r"michaelres.csv")

    df_combined = pd.DataFrame(columns=["Years of Experience", "Skills", "Location", "Designation"])
    df_combined = pd.concat([df_combined, job_desc, emp_df], axis=0)
    df_combined.reset_index(inplace=True, drop=True)
    df_combined = df_combined.drop(columns="Unnamed: 0")
    df_combined["Years of Experience"].fillna("Not Mentioned", inplace=True)
    df_combined["Designation"].fillna("Not Mentioned", inplace=True)
    df_combined["Skills"].fillna("Not Mentioned", inplace=True)
    df_combined["Location"].fillna("Not Mentioned", inplace=True)
    df_combined = df_combined.fillna(0)
    # print(df_combined)

    title = " "
    for i in emp_df.columns:
        title1 = emp_df[i]
        title = title + " " + str(title1[0])
        title = re.sub(r'[^a-zA-Z\s]', '', title)
    title = title.split(",")
    # print(title)
    claim = " "
    for i in job_desc.columns:
        claim1 = job_desc[i]
        claim = claim + " " + str(claim1[0])
        claim = re.sub(r'[^a-zA-Z\s]', '', claim)
    claim = claim.split(",")
    # print(claim)
    title = model.encode(title)
    claim = model.encode(claim)

    cos_sim = cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1))
    # print("Similarity: ", cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1)))

    cos = cos_sim[0]
    # print(cos)
    status = []
    # percent = int(input("Enter the threshold: "))
    if cos[0] > 0.8:
        sta = "Match"
        status.append(sta)
    else:
        sta = "Doesn't Match"
        status.append(sta)

    cosColumn_df = pd.DataFrame(columns=["Name", "Similarity", "Status"])
    cos_df = pd.DataFrame(dict, index=[0])
    # print(cos_df)
    cos_df = pd.concat([cosColumn_df, cos_df], axis=0)

    cos_df['Similarity'] = pd.Series(cos)
    cos_df['Status'] = pd.Series(status)
    cos_df["Name"].fillna("Not Mentioned", inplace=True)
    cos_df = cos_df[['Name', 'Similarity', 'Status']]
    # print(cos_df)

    # cosine_df.append(cos_df)
    cosine_df = pd.concat([cos_df, cosine_df], axis=0, ignore_index=True)
    print(cosine_df)


def customZip_ner(text):
    cosine_df = pd.DataFrame()
    for k in range(len(text)):
        dict = {}
        selected_dict = {}
        doc_ = nlp_ner(str(text[k]))
        for ent in doc_.ents:
            print(f'{ent.label_} - {ent.text}')
            dict.update({ent.label_: ent.text})
            if ent.label_ == "Years of Experience" or ent.label_ == "Skills" or ent.label_ == "Location" or ent.label_ == "Designation":
                selected_dict.update({ent.label_: ent.text})
        print("dict:", dict)
        print("selected_dict:", selected_dict)

        # creating dataframe for cos_sim

        # creating data frame with required features
        emp_df = pd.DataFrame(selected_dict, index=[0])
        print(emp_df)
        emp_df.to_csv("resumes_doc.csv")

        job_desc = pd.read_csv(r"michaelres.csv")

        df_combined = pd.DataFrame(columns=["Years of Experience", "Skills", "Location", "Designation"])
        df_combined = pd.concat([df_combined, job_desc, emp_df], axis=0)
        df_combined.reset_index(inplace=True, drop=True)
        df_combined = df_combined.drop(columns="Unnamed: 0")
        df_combined["Years of Experience"].fillna("Not Mentioned", inplace=True)
        df_combined["Designation"].fillna("Not Mentioned", inplace=True)
        df_combined["Skills"].fillna("Not Mentioned", inplace=True)
        df_combined["Location"].fillna("Not Mentioned", inplace=True)
        df_combined = df_combined.fillna(0)
        # print(df_combined)

        title = " "
        for i in emp_df.columns:
            title1 = emp_df[i]
            title = title + " " + str(title1[0])
            title = re.sub(r'[^a-zA-Z\s]', '', title)
        title = title.split(",")
        # print(title)
        claim = " "
        for i in job_desc.columns:
            claim1 = job_desc[i]
            claim = claim + " " + str(claim1[0])
            claim = re.sub(r'[^a-zA-Z\s]', '', claim)
        claim = claim.split(",")
        # print(claim)
        title = model.encode(title)
        claim = model.encode(claim)

        cos_sim = cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1))
        # print("Similarity: ", cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1)))

        cos = cos_sim[0]
        # print(cos)
        status = []
        if cos[0] > 0.8:
            sta = "Matches"
            status.append(sta)
        else:
            sta = "Doesn't Matches"
            status.append(sta)

        cosColumn_df = pd.DataFrame(columns=["Name", "Similarity", "Status"])
        cos_df = pd.DataFrame(dict, index=[0])
        cos_df = pd.concat([cosColumn_df, cos_df], axis=0)

        cos_df['Similarity'] = pd.Series(cos)
        cos_df['Status'] = pd.Series(status)
        cos_df["Name"].fillna("Not Mentioned", inplace=True)
        cos_df = cos_df[['Name', 'Similarity', 'Status']]
        # print(cos_df)

        # cosine_df.append(cos_df)
        cosine_df = pd.concat([cos_df, cosine_df], axis=0, ignore_index=True)
    print(cosine_df)


if __name__ == '__main__':
    app.run(debug=True)

