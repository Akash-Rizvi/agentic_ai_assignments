from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
import operator
from langgraph.graph import StateGraph, END

load_dotenv()

class SupportState(TypedDict):
    messages : Annotated[list[BaseMessage], operator.add]
    should_escalate : bool
    issue_type : str
    user_tier : str # "vip" or "standard"
 

model = ChatOpenAI(model="gpt-4.1-nano", seed=6, max_completion_tokens=100)   

def route_by_tier(state : SupportState) -> str:
    """Route based on user tier."""
    
    if state.get("user_tier") == "vip":
        return "vip_path"
    return "standard_path"

def check_user_tier_node(state: SupportState):
    """Decide if user is VIP or Standard (mock implementation)."""
    first_message = state["messages"][0].content.lower()
    if "vip" in first_message or "premium" in first_message:
        return{"user_tier":"vip"}
    return{"user_tier": "standard"}

def vip_agent_node(state: SupportState):
    """VIP path: fast lane, no escalation."""
    system_message = SystemMessage(content= "You are a VIP customer support agent. "
            "Respond politely, prioritize the user's issue, "
            "and keep the answer concise and helpful.")
    response = model.invoke([system_message]+state["messages"])
    return {"should_escalate": False,
            "messages":[response],}

def standard_agent_node(state: SupportState):
    """Standard path: may escalate."""
    system_message = SystemMessage(content="You are a standard customer support agent. "
            "Respond clearly and politely. If the issue may need human review, "
            "let the user know it can be escalated.")
    response = model.invoke([system_message] + state["messages"])
    return {"should_escalate": True,
            "messages": [response],}
    

def build_graph():
    workflow = StateGraph(SupportState)
    workflow.add_node("check_tier", check_user_tier_node)
    workflow.add_node("vip_agent", vip_agent_node)
    workflow.add_node("standard_agent", standard_agent_node)
    workflow.set_entry_point("check_tier")
    workflow.add_conditional_edges(
        "check_tier",
        route_by_tier,
        {"vip_path": "vip_agent",
         "standard_path": "standard_agent"}
    )
    workflow.add_edge("vip_agent", END)
    workflow.add_edge("standard_agent", END)
    return workflow.compile()
    


from langchain_core.messages import HumanMessage

def main () -> None:
    
    graph = build_graph()
    vip_result = graph.invoke({
        "messages" : [HumanMessage(content="I am a vip customer, please check my order")],
        "should_escalate" : False,
        "issue_type": "",
        "user_tier": "",
    })
    #print("VIP result:", vip_result.get("user_tier"), vip_result.get("should_escalate"))
    print("VIP response:", vip_result["messages"][-1].content)
    
    standard_result = graph.invoke({
        "messages" : [HumanMessage(content="Check my order status")],
        "should_escalate": False,
        "issue_type": "",
        "user_tier" : "",
    })
    #print("Standard results:", standard_result.get("user_tier"), standard_result.get("should_escalate"))
    print("Standard response:", standard_result["messages"][-1].content)
    
if __name__ == "__main__":
    main()