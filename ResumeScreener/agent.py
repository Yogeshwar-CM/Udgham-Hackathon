from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
import re
import PyPDF2

# Load environment variables
load_dotenv()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

# Define the evaluation criteria with thresholds
evaluation_criteria = """
Evaluate the candidate's resume based on the following criteria:
1. Job Match: Assess how well the candidate's experience and skills align with the job requirements. Score on a scale of 1 to 10.
2. Experience: Evaluate the relevance and depth of the candidate's professional experience. Score on a scale of 1 to 10.
3. Technical Skills: Analyze the proficiency and relevance of the technical skills listed. Score on a scale of 1 to 10.


Minimum acceptable scores:
- Job Match: 7
- Experience: 6
- Technical Skills: 8

A candidate must meet or exceed all minimum scores to be accepted. Provide a brief justification for each score and conclude with an overall recommendation: 'Accept' or 'Reject'.
"""

# Initialize the Groq model
model = Groq(id="llama-3.3-70b-versatile")

# Create the Agent
agent = Agent(
    model=model,
    description="An agent that evaluates resumes based on job match, experience, and technical skills.",
    instructions=evaluation_criteria,
    markdown=True
)

# Function to evaluate a resume
def evaluate_resume(resume_text):
    response = agent.run(resume_text)
    return response.content

# Function to extract scores and recommendation
def parse_evaluation(evaluation_text):
    scores = {
        'Job Match': None,
        'Experience': None,
        'Technical Skills': None
    }
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

# Define minimum acceptable scores
thresholds = {
    'Job Match': 7,
    'Experience': 6,
    'Technical Skills': 8
}

# Function to make the decision
def make_decision(scores, thresholds):
    for criterion, score in scores.items():
        if score is None or score < thresholds[criterion]:
            return 'Reject'
    return 'Accept'

# Specify the path to your resume PDF
pdf_path = "SampleResumes/Acsah.pdf"  # Change this to your actual file path

# Extract text from the PDF
resume_text = extract_text_from_pdf(pdf_path)

# Evaluate the resume
evaluation_result = evaluate_resume(resume_text)
print("\nEvaluation Result:\n", evaluation_result)

# Parse and analyze results
scores, recommendation = parse_evaluation(evaluation_result)
decision = make_decision(scores, thresholds)

print(f"\nFinal Decision: {decision}")
