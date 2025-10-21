from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import TypedDict

# 1. Define the model (Ollama must be running locally)
llm = ChatOllama(
    model="mistral",
    temperature=0,
    verbose=True,
)

# 2. Define the node behavior
def generate_response(state):
    prompt = state["input"]
    result = llm.invoke(prompt)
    return {"output": result.content}

# 3. Define the state schema using TypedDict
class State(TypedDict):
    input: str
    output: str

# 4. Build the graph
graph = StateGraph(State)
graph.add_node("chat", generate_response)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# 5. Compile it
app = graph.compile()

# 6. Run a simple test
if __name__ == "__main__":
    user_input = "Explain what the LangGraph framework is in one sentence."
    result = app.invoke({"input": user_input})
    print(result["output"])
