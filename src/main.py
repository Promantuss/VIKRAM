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
    # for page in doc:
    #     text = text + str(page.get_text())
    #     text = text.strip()
    #     text = text.replace("\n", " ")
    #     # keep only alphanumerics
    #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    #     text = text + " "
    #
    # custom_ner(text)

    list_test = ['Alice Clark  AI  Machine Learning Delhi India Email me on Indeed 20 years of experience in data handling design and development    Data Warehouse Data analysis starsnow flake scema data modelling and design specific to  data warehousing and business intelligence    Database Experience in database designing scalability backup and recovery writing and  optimizing SQL code and Stored Procedures creating functions views triggers and indexes  Cloud platform Worked on Microsoft Azure cloud services like Document DB SQL Azure  Stream Analytics Event hub Power BI Web Job Web App Power BI Azure data lake  analyticsUSQL  Willing to relocate anywhere    WORK EXPERIENCE  Software Engineer  Microsoft  Bangalore Karnataka  January 2000 to Present  1 Microsoft Rewards Live dashboards  Description  Microsoft rewards is loyalty program that rewards Users for browsing and shopping  online Microsoft Rewards members can earn points when searching with Bing browsing with  Microsoft Edge and making purchases at the Xbox Store the Windows Store and the Microsoft  Store Plus user can pick up bonus points for taking daily quizzes and tours on the Microsoft  rewards website Rewards live dashboards gives a live picture of usage worldwide and by  markets like US Canada Australia new user registration count topbottom performing rewards  offers orders stats and weekly trends of user activities orders and new user registrations the  PBI tiles gets refreshed in different frequencies starting from 5 seconds to 30 minutes  TechnologyTools used    EDUCATION  Indian Institute of Technology  Mumbai  2001    SKILLS  Machine Learning Natural Language Processing and Big Data Handling ADDITIONAL INFORMATION  Professional Skills   Excellent analytical problem solving communication knowledge transfer and interpersonal  skills with ability to interact with individuals at all the levels   Quick learner and maintains cordial relationship with project manager and team members and  good performer both in team and independent job environments   Positive attitude towards superiors amp peers   Supervised junior developers throughout project lifecycle and provided technical assistance ', 'Chethan  AI  Machine Learning Delhi India Email me on Indeed 20 years of experience in data handling design and development    Data Warehouse Data analysis starsnow flake scema data modelling and design specific to  data warehousing and business intelligence    Database Experience in database designing scalability backup and recovery writing and  optimizing SQL code and Stored Procedures creating functions views triggers and indexes  Cloud platform Worked on Microsoft Azure cloud services like Document DB SQL Azure  Stream Analytics Event hub Power BI Web Job Web App Power BI Azure data lake  analyticsUSQL  Willing to relocate anywhere    WORK EXPERIENCE  Software Engineer  Microsoft  Bangalore Karnataka  January 2000 to Present  1 Microsoft Rewards Live dashboards  Description  Microsoft rewards is loyalty program that rewards Users for browsing and shopping  online Microsoft Rewards members can earn points when searching with Bing browsing with  Microsoft Edge and making purchases at the Xbox Store the Windows Store and the Microsoft  Store Plus user can pick up bonus points for taking daily quizzes and tours on the Microsoft  rewards website Rewards live dashboards gives a live picture of usage worldwide and by  markets like US Canada Australia new user registration count topbottom performing rewards  offers orders stats and weekly trends of user activities orders and new user registrations the  PBI tiles gets refreshed in different frequencies starting from 5 seconds to 30 minutes  TechnologyTools used    EDUCATION  Indian Institute of Technology  Mumbai  2001    SKILLS  Machine Learning Natural Language Processing and Big Data Handling']
    custom_ner(list_test)


def custom_ner(text):
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
        # print(emp_df)
        emp_df.to_csv("resumes_doc.csv")

        job_desc = pd.read_csv(r"michaelres.csv")

        df_combined = pd.concat([job_desc, emp_df], axis=0)
        df_combined.reset_index(inplace=True, drop=True)
        df_combined = df_combined.drop(columns="Unnamed: 0")
        df_combined["Years of Experience"].fillna("Not Mentioned", inplace=True)
        df_combined["Location"].fillna("Not Mentioned", inplace=True)
        df_combined = df_combined.fillna(0)
        print(df_combined)
###############################################################################################
        title = " "
        for i in emp_df.columns:
            title1 = emp_df[i]
            title = title + " " + str(title1[0])
            title = re.sub(r'[^a-zA-Z\s]', '', title)
        title = title.split(",")
        print(title)
        claim = " "
        for i in job_desc.columns:
            claim1 = job_desc[i]
            claim = claim + " " + str(claim1[0])
            claim = re.sub(r'[^a-zA-Z\s]', '', claim)
        claim = claim.split(",")
        print(claim)
        title = model.encode(title)
        claim = model.encode(claim)

        cos_sim = cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1))
        print("Similarity: ", cosine_similarity(title[0].reshape(1, -1), claim[0].reshape(1, -1)))

##################################################################################################
        # a_list = []
        #
        # for i in range((df_combined.shape[0])):
        #     cur_row = []
        #     for j in range(df_combined.shape[1]):
        #         cur_row.append(df_combined.iat[i, j])
        #     a_list.append(cur_row)
        #
        # print("a_list:", a_list)
        # res = [' '.join(ele) for ele in a_list]
        # # print(res)
        #
        # title = model.encode(res)
        #
        # cos_sim = cosine_similarity([title[0]], title[1:])
        # print("Similarity: ", cosine_similarity([title[0]], title[0:]))

        cos = cos_sim[0]
        # print(cos)
        status = []
        if cos[0] > 0.8:
            sta = "Matches"
            status.append(sta)
        else:
            sta = "Doesn't Matches"
            status.append(sta)

        cos_df = pd.DataFrame(dict, index=[0])

        cos_df['Similarity'] = pd.Series(cos)
        cos_df['Status'] = pd.Series(status)

        cos_df = cos_df[['Name', 'Similarity', 'Status']]
        print(cos_df)

        # cosine_df.append(cos_df)
        cosine_df = pd.concat([cos_df, cosine_df], axis=0, ignore_index=True)
    print(cosine_df)


if __name__ == '__main__':
    app.run(debug=True)
