import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError


# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "banking_data.json"
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


# ============================================
# Data Loading and Formatting Functions
# ============================================
def load_data() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_inr(amount: float) -> str:
    return f"INR {amount:,.2f}"


# ============================================
# Transaction and Account Analysis Functions
# ============================================
def build_transactions_frame(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame(data["transactions"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


def total_balance(accounts: list[dict]) -> float:
    return sum(account["balance"] for account in accounts)


def debit_total(transactions: pd.DataFrame) -> float:
    debit_rows = transactions[transactions["type"] == "debit"]
    return float(debit_rows["amount"].sum())


def top_spending_category(transactions: pd.DataFrame) -> tuple[str, float]:
    debit_rows = transactions[transactions["type"] == "debit"]
    totals = debit_rows.groupby("category")["amount"].sum().sort_values(ascending=False)
    if totals.empty:
        return "None", 0.0
    return str(totals.index[0]), float(totals.iloc[0])


def spending_by_category(transactions: pd.DataFrame, category: str) -> float:
    debit_rows = transactions[transactions["type"] == "debit"]
    matches = debit_rows[debit_rows["category"].str.lower() == category.lower()]
    return float(matches["amount"].sum())


def saving_tip(transactions: pd.DataFrame) -> str:
    category, amount = top_spending_category(transactions)
    if amount == 0:
        return "Your recent transaction list is small, so I do not see a clear spending pattern yet."
    return (
        f"Your highest recent spending category is {category} at {format_inr(amount)}. "
        f"Reducing that category by even 10% could save about {format_inr(amount * 0.10)}."
    )


def build_banking_context(data: dict, transactions: pd.DataFrame) -> str:
    account_lines = []
    for account in data["accounts"]:
        account_lines.append(
            f"{account['type']} account {account['number']} balance {format_inr(account['balance'])}"
        )

    loan_lines = []
    for loan in data["loans"]:
        loan_lines.append(
            f"{loan['type']} outstanding {format_inr(loan['outstanding'])}, "
            f"EMI {format_inr(loan['emi'])}, next due {loan['next_due_date']}"
        )

    transaction_lines = []
    for _, row in transactions.head(8).iterrows():
        transaction_lines.append(
            f"{row['date'].date()} | {row['description']} | {row['category']} | "
            f"{row['type']} | {format_inr(row['amount'])}"
        )

    return "\n".join(
        [
            f"Customer: {data['customer']['name']}",
            f"Customer ID: {data['customer']['customer_id']}",
            f"Branch: {data['customer']['branch']}",
            f"Total balance: {format_inr(total_balance(data['accounts']))}",
            f"Recent debit spend: {format_inr(debit_total(transactions))}",
            "Accounts:",
            *account_lines,
            "Loans:",
            *loan_lines,
            "Recent transactions:",
            *transaction_lines,
        ]
    )


# ============================================
# Transaction Lookup Helpers
# ============================================
def ordinal_transaction_index(text: str) -> int | None:
    lookup = [
        ("second last transaction", 1),
        ("2nd last transaction", 1),
        ("third last transaction", 2),
        ("3rd last transaction", 2),
        ("fourth last transaction", 3),
        ("4th last transaction", 3),
        ("fifth last transaction", 4),
        ("5th last transaction", 4),
        ("last transaction", 0),
        ("latest transaction", 0),
    ]
    for phrase, index in lookup:
        if phrase in text:
            return index
    return None


def describe_transaction(transactions: pd.DataFrame, index: int) -> str:
    if index >= len(transactions):
        return "I do not have that many transactions in the current demo data."

    row = transactions.iloc[index]
    labels = {
        0: "latest",
        1: "second last",
        2: "third last",
        3: "fourth last",
        4: "fifth last",
    }
    prefix = labels.get(index, f"transaction number {index + 1} from the latest")
    return (
        f"Your {prefix} transaction is on {row['date'].date()}: "
        f"{row['description']} ({row['category']}) - "
        f"{row['type'].title()} {format_inr(row['amount'])}."
    )


# ============================================
# LLM Status Helpers
# ============================================
def set_llm_runtime_status(status: str, error_message: str = "") -> None:
    st.session_state["llm_runtime_status"] = status
    st.session_state["llm_runtime_error"] = error_message


def get_llm_runtime_status() -> tuple[str, str]:
    status = st.session_state.get("llm_runtime_status", "not_checked")
    error_message = st.session_state.get("llm_runtime_error", "")
    return status, error_message


# ============================================
# Rule-Based Assistant Response Logic
# ============================================
def answer_query_rule_based(query: str, data: dict, transactions: pd.DataFrame) -> str:
    text = query.lower().strip()
    accounts = data["accounts"]
    loans = data["loans"]

    transaction_index = ordinal_transaction_index(text)
    if transaction_index is not None:
        return describe_transaction(transactions, transaction_index)

    if (
        "debit" in text
        and any(word in text for word in ["spend", "spending", "recorded", "amount", "total"])
    ):
        return f"Your total recorded debit spending is {format_inr(debit_total(transactions))}."

    if any(word in text for word in ["balance", "total balance", "account balance"]):
        lines = [f"Your total balance is {format_inr(total_balance(accounts))}."]
        for account in accounts:
            lines.append(
                f"{account['type']} account ({account['number']}): {format_inr(account['balance'])}"
            )
        return "\n".join(lines)

    if "recent" in text or "last" in text or "transaction" in text:
        recent = transactions.head(5)
        lines = ["Here are your last 5 transactions:"]
        for _, row in recent.iterrows():
            lines.append(
                f"- {row['date'].date()}: {row['description']} ({row['category']}) - "
                f"{row['type'].title()} {format_inr(row['amount'])}"
            )
        return "\n".join(lines)

    if "food" in text:
        amount = spending_by_category(transactions, "Food")
        return f"You spent {format_inr(amount)} on food in the available transaction history."

    if "grocer" in text:
        amount = spending_by_category(transactions, "Groceries")
        return f"You spent {format_inr(amount)} on groceries in the available transaction history."

    if "spend" in text and "most" in text:
        category, amount = top_spending_category(transactions)
        return f"Your highest spending category is {category} with {format_inr(amount)} spent."

    if "save" in text or "saving tip" in text or "budget" in text:
        return saving_tip(transactions)

    if "loan" in text or "emi" in text:
        lines = ["Here is your loan summary:"]
        for loan in loans:
            lines.append(
                f"- {loan['type']}: outstanding {format_inr(loan['outstanding'])}, "
                f"EMI {format_inr(loan['emi'])}, next due {loan['next_due_date']}"
            )
        return "\n".join(lines)

    if "income" in text or "salary" in text:
        income_rows = transactions[transactions["type"] == "credit"]
        total_income = float(income_rows["amount"].sum())
        return f"Your total recorded income is {format_inr(total_income)}."

    if "summary" in text or "overview" in text:
        category, amount = top_spending_category(transactions)
        return (
            f"Account summary: total balance {format_inr(total_balance(accounts))}, "
            f"recent debit spending {format_inr(debit_total(transactions))}, "
            f"top spending category {category} at {format_inr(amount)}."
        )

    return (
        "I can help with balances, recent transactions, spending by category, savings tips, "
        "income summary, and loan details. Try asking: 'Show my balance' or "
        "'Where am I spending the most money?'"
    )


# ============================================
# LLM Configuration and API Functions
# ============================================
def get_llm_settings() -> dict:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    settings = {
        "provider": provider,
        "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant").strip(),
        "api_key": "",
    }

    if provider == "openai":
        settings["base_url"] = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        settings["model"] = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        settings["api_key"] = os.getenv("OPENAI_API_KEY", "").strip()
    else:
        settings["api_key"] = os.getenv("GROQ_API_KEY", "").strip()

    return settings


def llm_is_configured(settings: dict) -> bool:
    return bool(settings["api_key"] and settings["base_url"] and settings["model"])


def build_messages(
    query: str, data: dict, transactions: pd.DataFrame, chat_history: list[dict]
) -> list[dict]:
    system_prompt = (
        "You are a banking AI assistant for a demo application. "
        "Answer only from the provided banking data. "
        "If the user asks for something not present in the data, say that the information "
        "is not available in the current demo data. Keep answers concise, helpful, and factual."
    )
    context = build_banking_context(data, transactions)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Banking data:\n{context}"},
    ]

    recent_history = chat_history[-6:]
    for message in recent_history:
        if message["role"] in {"user", "assistant"}:
            messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": query})
    return messages


def call_llm_api(messages: list[dict], settings: dict) -> str:
    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
    try:
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=0.2,
        )
    except APIConnectionError as exc:
        raise RuntimeError(f"LLM API is unreachable: {exc}") from exc
    except RateLimitError as exc:
        raise RuntimeError(f"LLM API rate limit reached: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"LLM API request failed with HTTP {exc.status_code}: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM API returned an empty response.")
    return content.strip()


def generate_assistant_reply(
    query: str, data: dict, transactions: pd.DataFrame, chat_history: list[dict]
) -> tuple[str, str | None]:
    settings = get_llm_settings()
    if llm_is_configured(settings):
        set_llm_runtime_status("attempting")
        messages = build_messages(query, data, transactions, chat_history)
        try:
            response = call_llm_api(messages, settings)
            set_llm_runtime_status("success")
            return response, "llm"
        except RuntimeError as exc:
            set_llm_runtime_status("fallback_error", str(exc))
            st.warning(f"LLM mode failed, so the app used local banking logic instead. {exc}")
    else:
        set_llm_runtime_status("not_configured")

    return answer_query_rule_based(query, data, transactions), "rule-based"


# ============================================
# Session State Setup
# ============================================
def initialize_chat() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your Banking AI Assistant. "
                    "Ask me about balances, recent transactions, spending, or loans."
                ),
            }
        ]
    if "llm_runtime_status" not in st.session_state:
        set_llm_runtime_status("not_checked")


# ============================================
# Streamlit UI Components
# ============================================
def render_sidebar(data: dict, transactions: pd.DataFrame) -> None:
    accounts = data["accounts"]
    customer = data["customer"]
    spend_category, spend_amount = top_spending_category(transactions)
    settings = get_llm_settings()
    runtime_status, runtime_error = get_llm_runtime_status()

    st.sidebar.title("Customer Profile")
    st.sidebar.write(f"**Name:** {customer['name']}")
    st.sidebar.write(f"**Customer ID:** {customer['customer_id']}")
    st.sidebar.write(f"**Branch:** {customer['branch']}")
    st.sidebar.divider()
    st.sidebar.write(f"**Total Balance:** {format_inr(total_balance(accounts))}")
    st.sidebar.write(f"**Recent Debit Spend:** {format_inr(debit_total(transactions))}")
    st.sidebar.write(
        f"**Top Spending Category:** {spend_category} ({format_inr(spend_amount)})"
    )
    st.sidebar.divider()
    st.sidebar.subheader("Assistant Mode")
    st.sidebar.write(f"**Provider:** {settings['provider'].upper()}")
    st.sidebar.write(f"**Model:** {settings['model']}")
    if runtime_status == "success":
        st.sidebar.write("**Status:** LLM reply working")
    elif runtime_status == "fallback_error":
        st.sidebar.write("**Status:** Rule-based fallback after LLM error")
    elif llm_is_configured(settings):
        st.sidebar.write("**Status:** API key configured")
    else:
        st.sidebar.write("**Status:** Rule-based fallback")

    if runtime_error:
        st.sidebar.caption(f"Last LLM error: {runtime_error}")


def render_status_banner() -> None:
    runtime_status, runtime_error = get_llm_runtime_status()
    if runtime_status == "fallback_error":
        st.info(
            "LLM replies are currently unavailable, so the app is using local banking logic."
        )
        if runtime_error:
            st.caption(f"LLM error: {runtime_error}")


def render_dashboard(data: dict, transactions: pd.DataFrame) -> None:
    accounts = data["accounts"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Balance", format_inr(total_balance(accounts)))
    col2.metric("Debit Spending", format_inr(debit_total(transactions)))
    col3.metric("Transactions", len(transactions))

    st.subheader("Accounts")
    accounts_frame = pd.DataFrame(accounts)
    st.dataframe(accounts_frame, use_container_width=True, hide_index=True)

    st.subheader("Recent Transactions")
    st.dataframe(transactions.head(8), use_container_width=True, hide_index=True)


def submit_quick_query(query: str, data: dict, transactions: pd.DataFrame) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    response, source = generate_assistant_reply(query, data, transactions, st.session_state.messages)
    label = "LLM" if source == "llm" else "Rule-based"
    st.session_state.messages.append(
        {"role": "assistant", "content": f"{response}\n\n_Source: {label}_"}
    )


def render_quick_actions(data: dict, transactions: pd.DataFrame) -> None:
    st.subheader("Quick Actions")
    quick_queries = [
        "Show my balance",
        "Show my last 5 transactions",
        "How much did I spend on groceries?",
        "Where am I spending the most money?",
        "Give me saving tips",
        "What loans do I have?",
    ]

    cols = st.columns(3)
    for index, query in enumerate(quick_queries):
        if cols[index % 3].button(query, use_container_width=True):
            submit_quick_query(query, data, transactions)
            st.rerun()


def render_chat(data: dict, transactions: pd.DataFrame) -> None:
    st.subheader("Chat Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a banking question...")
    if prompt:
        submit_quick_query(prompt, data, transactions)
        st.rerun()


# ============================================
# Main Application Entry Point
# ============================================
def main() -> None:
    st.set_page_config(page_title="Banking AI Assistant", page_icon="bank", layout="wide")
    st.title("Banking AI Assistant")
    st.caption("A Streamlit banking assistant with optional real LLM API support.")

    data = load_data()
    transactions = build_transactions_frame(data)
    initialize_chat()

    render_sidebar(data, transactions)
    render_status_banner()
    render_dashboard(data, transactions)
    render_quick_actions(data, transactions)
    render_chat(data, transactions)


if __name__ == "__main__":
    main()
