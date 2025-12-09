import os
import PyPDF2
import spacy
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'resumes'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Job description (you can move it to a .txt file if needed)
job_description_text = """
We are seeking a data-driven and curious Data Scientist to join our analytics team. ...
"""

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join([page.extract_text() or '' for page in reader.pages])
    return text

@app.route('/')
def start():
    return render_template("start.html")

@app.route('/home', methods=['GET', 'POST'])
def home():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('resumes')
        resume_texts, filenames = [], []

        for file in files:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
            resume_texts.append(text)
            filenames.append(file.filename)

        # Vectorize and compute similarity
        vectorizer = CountVectorizer().fit([job_description_text] + resume_texts)
        vectors = vectorizer.transform([job_description_text] + resume_texts)
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        scores = [round(s * 100, 2) for s in cosine_sim]
        results = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(results, columns=["Resume", "Score"])
        df.to_csv(os.path.join(OUTPUT_FOLDER, "hr_report.csv"), index=False)

    return render_template("index.html", results=results)

@app.route('/download')
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, "hr_report.csv"), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
