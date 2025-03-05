import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import google.generativeai as genai
from PyPDF2 import PdfReader
import uvicorn
import io

# Load environment variables and configure API
genai.configure(api_key="AIzaSyCjNJ8fPtcYAFaPLGLGyZlFP2uOz62sdbA")
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(
    title="ResumeATS Pro API",
    description="API for Resume Analysis and ATS Optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_pdf(file):
    """
    Read PDF file and extract text
    """
    try:
        pdf_reader = PdfReader(file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        return pdf_text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def get_gemini_output(pdf_text, prompt):
    """
    Get Gemini AI output for resume analysis
    """
    try:
        response = model.generate_content([pdf_text, prompt])
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

@app.post("/quick-scan/")
async def quick_scan_resume(
    file: UploadFile = File(...), 
    job_description: Optional[str] = Form(None)
):
    """
    Quick Scan Resume Analysis Endpoint
    """
    # Read PDF file
    pdf_file = io.BytesIO(await file.read())
    pdf_text = read_pdf(pdf_file)

    # Prepare prompt
    prompt = f"""
    You are ResumeChecker, an expert in resume analysis. Provide a quick scan of the following resume:
    
    1. Identify the most suitable profession for this resume.
    2. List 3 key strengths of the resume.
    3. Suggest 2 quick improvements.
    4. Give an overall ATS score out of 100.
    
    Resume text: {pdf_text}
    Job description (if provided): {job_description}
    """

    # Generate analysis
    response = get_gemini_output(pdf_text, prompt)
    
    return {
        "analysis_type": "Quick Scan",
        "analysis_result": response
    }

@app.post("/detailed-analysis/")
async def detailed_analysis_resume(
    file: UploadFile = File(...), 
    job_description: Optional[str] = Form(None)
):
    """
    Detailed Resume Analysis Endpoint
    """
    # Read PDF file
    pdf_file = io.BytesIO(await file.read())
    pdf_text = read_pdf(pdf_file)

    # Prepare prompt
    prompt = f"""
    You are ResumeChecker, an expert in resume analysis. Provide a detailed analysis of the following resume:
    
    1. Identify the most suitable profession for this resume.
    2. List 5 strengths of the resume.
    3. Suggest 3-5 areas for improvement with specific recommendations.
    4. Rate the following aspects out of 10: Impact, Brevity, Style, Structure, Skills.
    5. Provide a brief review of each major section (e.g., Summary, Experience, Education).
    6. Give an overall ATS score out of 100 with a breakdown of the scoring.
    
    Resume text: {pdf_text}
    Job description (if provided): {job_description}
    """

    # Generate analysis
    response = get_gemini_output(pdf_text, prompt)
    
    return {
        "analysis_type": "Detailed Analysis",
        "analysis_result": response
    }

@app.post("/ats-optimization/")
async def ats_optimization_resume(
    file: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    ATS Optimization Resume Analysis Endpoint
    """
    # Read PDF file
    pdf_file = io.BytesIO(await file.read())
    pdf_text = read_pdf(pdf_file)

    # Prepare prompt
    prompt = f"""
    You are ResumeChecker, an expert in ATS optimization. Analyze the following resume and provide optimization suggestions:
    
    1. Identify keywords from the job description that should be included in the resume.
    2. Suggest reformatting or restructuring to improve ATS readability.
    3. Recommend changes to improve keyword density without keyword stuffing.
    4. Provide 3-5 bullet points on how to tailor this resume for the specific job description.
    5. Give an ATS compatibility score out of 100 and explain how to improve it.
    
    Resume text: {pdf_text}
    Job description: {job_description}
    """

    # Generate analysis
    response = get_gemini_output(pdf_text, prompt)
    
    return {
        "analysis_type": "ATS Optimization",
        "analysis_result": response
    }

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 8089))  # Default to 8089 if PORT is not set
     uvicorn.run(app, host="0.0.0.0", port=port)
