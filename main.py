from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from pydantic import BaseModel, Field
from typing import List
from rich.pretty import pprint
from agno.tools.duckduckgo import DuckDuckGoTools

class AgentResponseStructure(BaseModel):
    satisfied: bool = Field(default=False, description="If you have already asked 4 questions and gotten the answers from the candidate as well, only then return True.")
    agentResponse: str = Field(description="Short response for the prompt the candidate gives")

tech_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions="you are a tech interviewer, speak like one. You will be interviewing a candidate for a tech role.",
    expected_output="Give output in an oral conversational manner. Avoid using long sentences.",
    add_history_to_messages=True,
    num_history_responses= 5,
    response_model=AgentResponseStructure,
    markdown=True
)

soft_skills_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="I am the Soft Skills Interviewing Bot. I evaluate communication and leadership skills.",
    show_tool_calls=True,
    response_model=AgentResponseStructure,
    markdown=True
)

cult_fit_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    response_model=AgentResponseStructure,
    description="I am the Cultural Fit Interviewing Bot. Let's see if you align with our company culture.",
    markdown=True
)

aptitude_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
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
