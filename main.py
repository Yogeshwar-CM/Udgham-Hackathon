from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

# Define all four agents
tech_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a Tech Interviewing Bot conducting a technical interview. Maintain a professional and authoritative tone. Always stay in character and address the candidate with respect.",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

soft_skills_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="I am the Cultural Fit Interviewing Bot. Let's see if you align with our company culture.",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

cult_fit_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a Cultural Fit Interviewing Bot evaluating alignment with company values. Keep a professional and authoritative demeanor. Stay in character and interact with the candidate accordingly.",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

aptitude_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are an Aptitude Interviewing Bot testing logical reasoning and problem-solving skills. Maintain a formal and authoritative tone. Always remain in character during the interview.",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# List of agents
agents = [tech_agent, soft_skills_agent, cult_fit_agent, aptitude_agent]

def interview_workflow():
    print("\nWelcome to the Multi-Agent Interview System!\n")
    total_questions_per_agent = 5

    for agent in agents:
        print(f"\n{agent.description}\n")
        
        question_count = 0
        while question_count < total_questions_per_agent:
            # Agent asks a question
            agent.print_response("Ask the next interview question.", stream=True)
            question_count += 1
            
            while True:
                user_input = input("\nYour response: ")

                if user_input.lower() in ["exit", "quit"]:
                    print("\nExiting interview...")
                    return
                
                # Agent processes the response and decides whether to follow up or proceed
                agent.print_response(user_input, stream=True)
                
                follow_up = input("\nDo you have a follow-up question? (yes/no): ").strip().lower()
                if follow_up == "no":
                    break

    print("\nInterview completed. Thank you!\n")

if __name__ == "__main__":
    interview_workflow()