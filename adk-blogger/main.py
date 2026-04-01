import asyncio
from agent import root_agent


async def main():
    topic = "The Future of AI Agents in Software Engineering"
    print(f"Running agent for topic: {topic}")
    # The run method is usually how you call an ADK agent
    # Depending on the ADK version, it might be run() or something similar
    # Here we assume a simple run call
    try:
        response = await root_agent.run(f"Write a blog post about {topic}")
        print("
--- Final Blog Post ---
")
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
