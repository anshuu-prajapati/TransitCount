# Agentic SDLC - Setup Guide

You don't need any model installed locally. This uses **Groq's free API**, which
hosts Llama 3.3-70B and Qwen models in the cloud — you just call them over the internet.

## Step 1: Get a free Groq API key
1. Go to https://console.groq.com/keys
2. Sign up (free, no credit card needed)
3. Click "Create API Key", copy it

## Step 2: Install Python dependencies
Open a terminal in this folder and run:
```
pip install -r requirements.txt
```

## Step 3: Add your API key
1. Rename `.env.example` to `.env`
2. Open `.env` and paste your key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

## Step 4: Run it
```
python main.py
```

That's it. From here everything is automatic:
1. Agent 1 writes requirements to `.vscode/requirements.md`
2. Agent 2 writes a spec to `.vscode/architecture_spec.md`
3. Agent 3 scores the spec — if below 75%, it sends feedback back to Agent 2 (loops, up to 4 times)
4. Agent 4 writes the UI code to `src/index.html`
5. Agent 5 scores the code — if below 75%, it sends feedback back to Agent 4 (loops, up to 4 times)
6. Agent 6 starts a local server and opens your browser automatically showing the finished "Hello Rushikesh" UI

Watch the terminal — every agent prints what it's doing and its score, so you can see
the conversation happening in real time.

## Changing the request
To build something other than "Hello Rushikesh", edit this line near the top of `main.py`:
```python
USER_GOAL = "Create a 'Hello Rushikesh' UI with an advanced, modern, animated look using HTML/CSS/JS."
```

## Notes
- `MAX_LOOPS = 4` caps the feedback loops so it can't run forever if a score never
  reaches 75%. Increase it if you want more refinement attempts.
- `ACCURACY_THRESHOLD = 75` matches your diagram — change it if you want a stricter/looser bar.
- Groq's free tier has rate limits (requests/day). If you hit them, wait a bit or
  reduce `MAX_LOOPS`.
- Everything else (Agent 6, the server) runs locally on your machine — only the
  Agent 1-5 LLM calls go over the internet to Groq.
