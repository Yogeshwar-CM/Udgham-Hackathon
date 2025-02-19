from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from pydantic import BaseModel, Field
import PyPDF2
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from typing import List
from rich.pretty import pprint
from agno.tools.duckduckgo import DuckDuckGoTools

# pdf_knowledge_base = PDFKnowledgeBase(
#     path="job_requirements",  
#     reader=PDFReader(chunk=True),
# )

# # Load the knowledge base
# pdf_knowledge_base.load(recreate=False)

class AgentResponseStructure(BaseModel):
    satisfied: bool = Field(default=False, description="If you have already asked 3 questions and gotten the answers from the candidate as well, only then return True.")
    agentResponse: str = Field(description="Short response for the prompt the candidate gives")

tech_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    # knowledge=pdf_knowledge_base,
    instructions="i want you to ask questions like 'what do you know about something' because remember that its like a human conversation and ask them single line questions on what they know, and give them a happy response if its a good answer and be disappointed if it makes no sense. The questions must strictly be related to the tech interview, DONT GO OUT OF TOPIC",
    expected_output="Give output in an oral conversational manner. Avoid using long sentences.",
    add_history_to_messages=True,
    num_history_responses= 5,
    response_model=AgentResponseStructure,
    markdown=True
)

soft_skills_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="I am the Soft Skills Interviewing Bot. I test communication and interpersonal skills.",
    add_history_to_messages=True,
    num_history_responses= 5,
    instructions="i want you to ask questions in an interactive and constructive method but dont overexplain things, just be concise. Include appreciatation when needed",
    # knowledge=pdf_knowledge_base,
    response_model=AgentResponseStructure,
    markdown=True
)

cult_fit_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    add_history_to_messages=True,
    instructions="i want you to ask questions in an interactive and constructive method but dont overexplain things, just be concise. Check if their values resonates with the company's mission. make it enthusiastic and give sarcastic responses. Dont repeat the questions.",
    num_history_responses= 5,
    # knowledge=pdf_knowledge_base,
    response_model=AgentResponseStructure,
    description="I am the Cultural Fit Interviewing Bot. Let's see if you align with our company culture.",
    markdown=True
)

aptitude_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    add_history_to_messages=True,
    num_history_responses= 5,
    # knowledge=pdf_knowledge_base,
    response_model=AgentResponseStructure,
    description="I am the Aptitude Interviewing Bot. I test logical reasoning and problem-solving skills.",
    markdown=True
)

agents = [tech_agent, soft_skills_agent, cult_fit_agent, aptitude_agent]

def chat_with_agent(agent):
    while True:
        user_message = input("\nYour response: ")
        response = agent.run(user_message)
        print("\nAgent:", response.content.agentResponse)
        
        if response.content.satisfied:
            print("\nAgent is satisfied. Moving to the next section...")
            break  

def interview_workflow():
    print("\nWelcome to the Multi-Agent Interview System!\n")

    for agent in agents:
        print('Interview Section')
        chat_with_agent(agent)

    print("\nInterview completed. Thank you!\n")

if __name__ == "__main__":
    interview_workflow()
