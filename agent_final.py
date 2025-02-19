from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from dotenv import load_dotenv
import os
import re
import PyPDF2

# Load environment variables
load_dotenv()

# Initialize PDF Knowledge Base (without using a database)
pdf_knowledge_base = PDFKnowledgeBase(
    path="job_requirements",  # Folder containing job requirement PDFs
    reader=PDFReader(chunk=True),
)

# Load the knowledge base
pdf_knowledge_base.load(recreate=False)

# Initialize the Groq model
model = Groq(id="llama-3.3-70b-versatile")

# Create the Agent with knowledge base
agent = Agent(
    model=model,
    knowledge=pdf_knowledge_base,
    search_knowledge=True,
    description="An agent that evaluates resumes based on job requirements, experience, and technical skills.",
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

# Function to extract email from resume text
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

# Function to evaluate a resume against job requirements
def evaluate_resume(resume_text, job_requirements):
    prompt = f"""
    Evaluate the candidate's resume against the following job requirements:

    {job_requirements}

    Provide a score for the following categories:
    1. Job Match: (1-10)
    2. Experience: (1-10)
    3. Technical Skills: (1-10)

    A candidate must meet or exceed these thresholds:
    - Job Match: 7
    - Experience: 6
    - Technical Skills: 8

    Provide a brief justification for each score and conclude with a line starting with 'Recommendation:' followed by 'Accept' or 'Reject'.
    """

    response = agent.run(prompt + "\n\nResume:\n" + resume_text)
    return response.content

# Function to extract scores and recommendation
def parse_evaluation(evaluation_text):
    scores = {'Job Match': None, 'Experience': None, 'Technical Skills': None}
    recommendation = None

    # Regular expressions to extract scores and recommendation
    score_patterns = {
        'Job Match': r'Job Match:\s*([0-9]+)',
        'Experience': r'Experience:\s*([0-9]+)',
        'Technical Skills': r'Technical Skills:\s*([0-9]+)'
    }
    recommendation_pattern = r'Recommendation:\s*(Accept|Reject)'

    for key, pattern in score_patterns.items():
        match = re.search(pattern, evaluation_text)
        if match:
            scores[key] = int(match.group(1))

    recommendation_match = re.search(recommendation_pattern, evaluation_text)
    if recommendation_match:
        recommendation = recommendation_match.group(1)

    return scores, recommendation

# Load job requirements from PDF
job_requirements_pdf = "job_requirements/job_desc.pdf"  # Change as needed
job_requirements_text = extract_text_from_pdf(job_requirements_pdf)

# Path to resumes folder
resumes_folder = "SampleResumes/"  # Change as needed
resume_files = [f for f in os.listdir(resumes_folder) if f.endswith(".pdf")]

# Process each resume
for resume_file in resume_files:
    resume_path = os.path.join(resumes_folder, resume_file)
    resume_text = extract_text_from_pdf(resume_path)

    # Extract email
    email = extract_email(resume_text)

    # Evaluate resume
    evaluation_result = evaluate_resume(resume_text, job_requirements_text)
    print(f"\nEvaluation Result for {resume_file}:\n", evaluation_result)

    # Parse results
    scores, recommendation = parse_evaluation(evaluation_result)
    decision = recommendation if recommendation else "Reject"

    print(f"\nFinal Decision for {resume_file}: {decision}")
