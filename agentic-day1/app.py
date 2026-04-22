from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4.1-nano", seed=6, max_completion_tokens=100)

resp1 = model.invoke("We are building an AI system for processing medical insurance claims.")
resp2 = model.invoke("What are the main risks in this system?")

print("Response 1:")
print(resp1.content)

print("\nResponse 2:")
print(resp2.content)

# Why the second question may fail or behave inconsistently without conversation history.

# Ans: The model is stateless. It cannot remember the earlier conversation that happened between User and AI

# Message-based invocation
messages = [SystemMessage(content="You are a senior AI architect reviewing production systems."),
            HumanMessage(content="We are building an AI system for processing medical insurance claims."),
            HumanMessage(content="What are the main risks in this system?")]

resp3= model.invoke(messages)

print("\nMessage-based Response:")
print(resp3.content)

# Reflection-break
print("\nReflection:")

print("1. Why did string-based invocation fail?")
print("String-based invocation failed because the model is stateless and does not retain previous context between calls.")

print("\n2. Why does message-based invocation work?")
print("Message-based invocation works because it includes the full conversation history, allowing the model to understand context.")

print("\n3. What would break in a production AI system if we ignore message history?")
print("Ignoring message history would break multi-turn conversations, reduce response accuracy, and lead to inconsistent and contextless outputs.")