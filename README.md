# Banking AI Assistant

A simple Streamlit-based Banking AI Assistant built for assignment/demo use, now with optional real LLM API support.

## What it does

- Shows a customer banking dashboard
- Answers common banking questions in a chat interface
- Explains account balances
- Summarizes recent transactions
- Gives simple budget and spending insights
- Can use a real LLM API for more natural answers
- Falls back to local rule-based banking logic if no API key is configured
- Supports quick actions like:
  - `Show my balance`
  - `Recent transactions`
  - `How much did I spend on food?`
  - `Give me saving tips`

## Project Structure

```text
Bank_AI_Assistant/
|-- app.py
|-- banking_data.json
|-- .env.example
|-- requirements.txt
|-- README.md
```

## Run Locally

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Optional: configure an LLM API

Create a `.env` file in the project folder and copy values from `.env.example`.

Example for Groq:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
LLM_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=your_groq_api_key_here
```

Example for OpenAI:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key_here
```

4. Start the app:

```powershell
streamlit run app.py
```

## Sample Questions

- `What is my total balance?`
- `Show my last 5 transactions`
- `How much did I spend on groceries?`
- `Where am I spending the most money?`
- `Can I save more this month?`
- `What loans do I have?`

## Notes

- This project uses sample banking data for demonstration.
- No real banking integration is included.
- The assistant answers from the sample banking data shown in the app.
- If the LLM is not configured, the app still works in rule-based mode.
- The project is intentionally simple and easy to explain in an assignment/demo.
