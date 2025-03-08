import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import docx2txt
import plotly.express as px

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(BytesIO(file))
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        return docx2txt.process(BytesIO(file))
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0].reshape(1, -1)
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()
    return cosine_similarities * 100

# Streamlit UI
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description here:")

# File uploader for resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("📊 Ranking Resumes")
    
    resumes = []
    file_names = []
    
    for file in uploaded_files:
        file_names.append(file.name)
        file_bytes = file.getvalue()
        
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_docx(file_bytes)
        
        resumes.append(text)
    
    scores = rank_resumes(job_description, resumes)
    
    # Create and display results
    results = pd.DataFrame({"Resume": file_names, "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
    
    # Display top-ranked resume
    best_resume = results.iloc[0]['Resume']
    st.subheader(f"🥇 Best Matched Resume: {best_resume}")
    st.write(f"*Score:* {results.iloc[0]['Score']:.2f}")
    
    # Resume preview
    if st.checkbox("Show Best Resume Content"):
        best_resume_index = file_names.index(best_resume)
        st.write(resumes[best_resume_index])
    
    # Visualization
    fig = px.bar(results, x='Resume', y='Score', title="Resume Ranking Scores", color='Score', labels={'Score': 'Matching Score'})
    st.plotly_chart(fig)
    
    # Download options
    st.subheader("📥 Download Ranked Resumes")
    file_format = st.radio("Choose file format:", ("CSV", "Text"))
    
    if file_format == "CSV":
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "ranked_resumes.csv", "text/csv")
    else:
        text_data = "\n".join([f"{row.Resume}: {row.Score:.2f}" for _, row in results.iterrows()])
        st.download_button("Download Text File", text_data, "ranked_resumes.txt", "text/plain")