import pandas as pd
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ========================================
# 0. Setup Azure Entra ID Token Provider
# ========================================

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://management.azure.com/.default"
)

MS_FOUNDRY_ENDPOINT="https://jdbacolod-devo-6315-resource.services.ai.azure.com/openai/v1"
api_key="9C6N38dIxBhBUZBuA6JzV2rIOTP6nu4zYkSj7i7FrFsLYXAdEq60JQQJ99CIAC5RqLJXJ3w3AAAAACOGFnKo"

foundry_gpt_mini = ChatOpenAI(
    model="gpt-5-mini",
    base_url=MS_FOUNDRY_ENDPOINT,
    api_key=api_key,
    temperature=0.0
)

# ============================
# 1. State and Output Schema
# ============================

class BinaryClassificationResult(BaseModel):
    reasoning: str = Field(description="Brief explanation of why the text matches or or does not match.")
    is_match: bool = Field(description="True if text matches the category, False otherwise.")

class WaterfallState(TypedDict):
    text: str
    assigned_category: Optional[str]
    winning_reasoning: Optional[str]

# Setup prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a text classifier. Determine if the text belongs strictly to this category.\n"
     "Category: {category_name}\n"
     "Category Description: {category_description}"),
     ("human", "Text: {text}")
])

CATEGORIES = [
    {
        "id": "1",
        "name": "Contract",
        "description": "All requests, incidents, or questions about employment contracts: creation, modification, cancellation, signing, or exceptional situations. Often priority 1 and must be treated urgently.",
        "model": foundry_gpt_mini
    },
    {
        "id": "2",
        "name": "Social documents",
        "description": "Administrative employment-related documents for candidates/clients, including requests, uploads, and clarifications, typically retrievable via Mytools or uploaded by candidates.",
        "model": foundry_gpt_mini
    },
    {
        "id": "3",
        "name": "Profile",
        "description": "Anything related to candidate or client user profiles in the app/desktop, from account creation and access to profile validation and attributes.",
        "model": foundry_gpt_mini
    },
]

FALLBACK_CATEGORY = "Other_Unclassified"

# ================================
# 2. Node and Router Definitions
# ================================

def make_classifier_node(cat_info: dict):
    """Factory creating a unique graph node for each category classifier."""
    def node_fn(state: WaterfallState) -> WaterfallState:
        # If already classified by an earlier node, skip evaluation
        if state.get("assigned_category"):
            return state

        # Bind Pydantic output schema to model
        llm = cat_info["model"].with_structured_output(BinaryClassificationResult)
        chain = prompt_template | llm

        result: BinaryClassificationResult = chain.invoke({
            "category_name": cat_info["name"],
            "category_description": cat_info["description"],
            "text": state["text"]
        })

        if result.is_match:
            return {
                **state,
                "assigned_category": cat_info["id"],
                "winning_reasoning": result.reasoning
            }

        return state

    return node_fn

def fallback_node(state: WaterfallState) -> WaterfallState:
    """Invoked only if all preceding classifiers return False."""
    if not state.get("assigned_category"):
        return {
            **state,
            "assigned_category": FALLBACK_CATEGORY,
            "winning_reasoning": "Rejected by all 3 specific classifiers. Categorized as Fallback."
        }
    return state

def check_short_circuit(state: WaterfallState) -> str:
    """Router decision: If category found, stop graph early. Else move to next classifier."""
    if state.get("assigned_category"):
        return "stop"
    return "continue"

# =====================
# 3. Graph Assembly
# =====================

builder = StateGraph(WaterfallState)

# Add dynamic classifier nodes
node_names = []
for cat in CATEGORIES:
    node_name = f"eval_{cat['id']}"
    node_names.append(node_name)
    builder.add_node(node_name, make_classifier_node(cat))

# Add fallback node
builder.add_node("fallback", fallback_node)

# Build sequential edges with short-circuiting routing
builder.add_edge(START, node_names[0])

for i in range(len(node_names)):
    current_node = node_names[i]
    next_node = node_names[i + 1] if i + 1 < len(node_names) else "fallback"

    builder.add_conditional_edges(
        current_node,
        check_short_circuit,
        {
            "stop": END,            # Category matched: EXIT IMMEDIATELY (Save API calls & time)
            "continue": next_node   # No match: PASS TO THE NEXT CLASSIFIER
        }
    )

builder.add_edge("fallback", END)
app = builder.compile()

# ==========================
# 4. CSV Batch Processing
# ==========================

def process_csv_cascade(csv_path: str, text_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    assigned_categories = []
    reasonings = []

    print("--- Starting Cascade Binary Classification Processing ---")

    for idx, row in df.iterrows():
        input_text = str(row[text_column])

        initial_state: WaterfallState = {
            "text": input_text,
            "assigned_category": None,
            "winning_reasoning": None
        }

        # Runs through graph until matched or fallen back
        final_state = app.invoke(initial_state)

        assigned_categories.append(final_state["assigned_category"])
        reasonings.append(final_state["winning_reasoning"])

        print(f"Row {idx+1} | Category: [{final_state['assigned_category']}] | Text: '{input_text[:35]}...'")

    df["final_assigned_category"] = assigned_categories
    df["category_reasoning"] = reasonings

    return df

# ========================
# 5. Example Execution
# ========================
def main():
    data = {
        "document_id": [101, 102, 103, 104],
        "corpus_text": [
            "Le présent contrat peut être résilié par chacune des parties moyennant un préavis de trois mois.",
            "Client agrees to pay all monthly fees within 30 days of invoice receipt.",
            "De werkgever is verplicht om de werknemer te voorzien van een veilige werkomgeving.",
            "The weather in Brussels tomorrow will be mostly cloudy with occasional rain showers."
        ]
    }

    csv_file = "contracts_test.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)

    result_df = process_csv_cascade(csv_file, text_column="corpus_text")

    print("\n--- Final Results ---")
    print(result_df[["document_id", "final_assigned_category", "category_reasoning"]])    


if __name__ == "__main__":
    main()
