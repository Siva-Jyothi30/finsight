"""
agent.py - FinSight Personal Finance Analytics Agent
=====================================================
Builds a conversational AI agent that answers natural-language questions
about the user's transaction data. The agent uses:

  - LangChain 1.x  : LCEL (LangChain Expression Language) runnable chain
  - Groq            : fast LLM inference (llama-3.3-70b-versatile)
  - python-dotenv   : loads GROQ_API_KEY from the .env file

Architecture
------------
1. ``build_context_summary()``
   Computes a compact text summary of the DataFrame (total spend,
   per-category breakdown, top merchants, behavioral patterns, etc.)
   This is baked into the system prompt so the model answers from facts.

2. ``build_agent()``
   Creates a LangChain LCEL chain:  prompt | llm | output_parser
   Returns a FinSightAgent wrapper that holds the chain + chat history.

3. ``FinSightAgent.ask()``
   Public entry point. Appends the human message to history, calls
   the chain, appends the AI reply, and returns the response string.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env (GROQ_API_KEY)
load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL   = "llama-3.3-70b-versatile"   # best free-tier Groq model (2025)
MEMORY_TURNS = 10                            # remember last 10 conversation turns
TEMPERATURE  = 0.4                           # balanced creativity vs. factuality


# ---------------------------------------------------------------------------
# 1. Context builder: converts the DataFrame into a text summary for the model
# ---------------------------------------------------------------------------

def build_context_summary(df: pd.DataFrame) -> str:
    """
    Generate a structured text summary of the transaction DataFrame.

    This summary is injected into the system prompt so the model has
    accurate, up-to-date statistics without needing database access.

    Parameters
    ----------
    df : pd.DataFrame
        Fully loaded and categorized DataFrame (output of categorize()).

    Returns
    -------
    str
        A multi-section plain-text summary of the user's finances.
    """
    total_spend   = df["amount"].sum()
    total_txns    = len(df)
    date_min      = df["datetime"].min().strftime("%B %d, %Y")
    date_max      = df["datetime"].max().strftime("%B %d, %Y")
    avg_txn       = df["amount"].mean()
    max_txn       = df["amount"].max()
    fraud_count   = int(df["is_fraud"].sum())
    fraud_pct     = 100 * fraud_count / total_txns

    # Per-category breakdown
    cat_summary = (
        df.groupby("spending_category", observed=True)["amount"]
        .agg(total="sum", count="count")
        .reset_index()
        .sort_values("total", ascending=False)
    )
    cat_lines = [
        f"  • {row['spending_category']:<22} "
        f"${row['total']:>12,.2f}  "
        f"({100 * row['total'] / total_spend:.1f}%)  "
        f"{int(row['count']):>6,} transactions"
        for _, row in cat_summary.iterrows()
    ]

    # Top 10 merchants by spend
    top_merchants = (
        df.groupby("merchant", observed=True)["amount"]
        .sum().sort_values(ascending=False).head(10)
    )
    merch_lines = [
        f"  {i+1:>2}. {name:<40} ${spend:>10,.2f}"
        for i, (name, spend) in enumerate(top_merchants.items())
    ]

    # Last 6 months of monthly spend
    monthly = (
        df.groupby(["year", "month", "month_name"], observed=True)["amount"]
        .sum().reset_index()
        .sort_values(["year", "month"]).tail(6)
    )
    monthly_lines = [
        f"  • {row['month_name']:<12}  ${row['amount']:>10,.2f}"
        for _, row in monthly.iterrows()
    ]

    # Behavioral peaks
    peak_day  = df.groupby("day_of_week")["amount"].sum().idxmax()
    peak_hour = df.groupby("hour")["amount"].sum().idxmax()

    summary = f"""=== FINSIGHT DATA SUMMARY ===

OVERVIEW
  Date range      : {date_min} → {date_max}
  Total spend     : ${total_spend:,.2f}
  Total txns      : {total_txns:,}
  Avg transaction : ${avg_txn:.2f}
  Largest txn     : ${max_txn:,.2f}
  Fraud txns      : {fraud_count:,} ({fraud_pct:.2f}% of all transactions)

SPENDING BY CATEGORY
{chr(10).join(cat_lines)}

TOP 10 MERCHANTS BY SPEND
{chr(10).join(merch_lines)}

RECENT MONTHLY SPENDING (last 6 months)
{chr(10).join(monthly_lines)}

BEHAVIORAL PATTERNS
  Peak spending day  : {peak_day}
  Peak spending hour : {peak_hour:02d}:00"""

    return summary


# ---------------------------------------------------------------------------
# 2. Agent: wraps the LCEL chain with in-memory conversation history
# ---------------------------------------------------------------------------

@dataclass
class FinSightAgent:
    """
    Stateful wrapper around a LangChain LCEL chain.

    Maintains a rolling window of conversation history so the model
    understands follow-up questions in context.

    Attributes
    ----------
    chain : langchain_core.runnables.Runnable
        The compiled runnable chain: prompt | model | parser.
    history : list[BaseMessage]
        Chronological list of Human and AI messages (the conversation log).
    max_turns : int
        Max number of FULL turns (human + AI pairs) to keep in memory.
    """
    chain     : object                       # Runnable (not typed to avoid import complexity)
    history   : list[BaseMessage] = field(default_factory=list)
    max_turns : int = MEMORY_TURNS

    def ask(self, question: str) -> str:
        """
        Send a question to the FinSight agent and return its response.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        str
            The agent's response as a plain string.
        """
        if not question.strip():
            return "Please ask me something about your finances!"

        # Trim history to the rolling window BEFORE invoking the chain
        # Each turn = 1 HumanMessage + 1 AIMessage = 2 messages
        max_messages = self.max_turns * 2
        trimmed_history = self.history[-max_messages:] if len(self.history) > max_messages else self.history

        # Call the chain
        response: str = self.chain.invoke({
            "history": trimmed_history,
            "input":   question,
        })

        # Persist this turn to history
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=response))

        return response

    def clear_history(self) -> None:
        """Reset the conversation history (start a fresh session)."""
        self.history.clear()


def build_agent(context_summary: str) -> FinSightAgent:
    """
    Build and return a FinSightAgent powered by Groq via LCEL.

    The system prompt embeds the data context summary so the model
    answers from computed facts rather than inventing numbers.

    Parameters
    ----------
    context_summary : str
        Output of ``build_context_summary()``. Injected as ground-truth
        financial data into the system prompt.

    Returns
    -------
    FinSightAgent
        A stateful agent instance ready to answer questions.

    Raises
    ------
    ValueError
        If ``GROQ_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here"
        )

    # Initialise the Groq chat model
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        api_key=api_key,
    )

    # Build the system prompt with the data summary embedded inside it.
    # MessagesPlaceholder injects the rolling history list at runtime.
    system_prompt = (
        "You are FinSight, a sharp and friendly personal finance analyst.\n"
        "You have been given a complete summary of the user's transaction data below.\n"
        "Use ONLY this data when answering factual questions. Never guess or make up numbers.\n\n"
        "When explaining trends, speak like a financial storyteller: "
        "be concise, insightful, and occasionally point out surprising patterns.\n"
        "If the user asks something the data cannot answer, say so honestly.\n\n"
        f"--- DATA SUMMARY ---\n{context_summary}\n--- END OF DATA SUMMARY ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Build the runnable chain: prompt → model → plain string
    chain = prompt | llm | StrOutputParser()

    return FinSightAgent(chain=chain)


# ---------------------------------------------------------------------------
# Quick self-test: 3-turn terminal conversation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data
    from categorizer import categorize

    print("Loading data ...")
    df = load_data(sample_size=100_000)
    df = categorize(df)

    print("Building context summary ...")
    context = build_context_summary(df)
    print(context)
    print()

    print("Building agent ...")
    agent = build_agent(context)

    test_questions = [
        "What is my total spending and how many transactions are there?",
        "Which category takes the biggest bite out of my budget, and by how much?",
        "When do I tend to spend the most? Which day and hour?",
    ]

    print("=" * 60)
    print("FinSight Agent: Test Conversation")
    print("=" * 60)
    for q in test_questions:
        print(f"\nUser : {q}")
        answer = agent.ask(q)
        print(f"Agent: {answer}")

    print("\nAgent test complete.")
