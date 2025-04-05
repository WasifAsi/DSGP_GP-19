# 🌊 Shoreline Analysis Chatbot – Powered by Rasa

This is an AI-powered chatbot designed to answer questions related to shoreline analysis, including **NSM (Net Shoreline Movement)**, **EPR (End Point Rate)**, and **coastal erosion**. Built using the **Rasa** framework, it's easy to train, extend, and integrate with frontend applications.

---

## 🧰 Prerequisites

- Python 3.8 or 3.9
- pip (Python package manager)
- Git (to clone the repository)

---

## 📥 Step 1: Install Rasa

### 1. (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 2. Upgrade pip and install Rasa

```bash
pip install --upgrade pip
pip3 install uv
uv pip install rasa-pro
```

> If installation fails, try:  
> `pip install rasa==3.6.10`

---

## 📦 Step 2: Clone the Repository

```bash
git clone https://github.com/yourusername/shoreline-chatbot.git
cd shoreline-chatbot
```

---

## 🔧 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, manually install:

```bash
pip install rasa-sdk
```

---

### Option 4: Run with custom actions

In one terminal:

```bash
rasa run actions
```

In another terminal:

```bash
rasa run --enable-api --cors "*" --debug
```

---

## 🖥️ Step 5: Chat with the Bot (Terminal)

```bash
rasa shell
```

---

## 🌐 Step 6: Integrate via REST API

You can send messages to the bot using:

```
POST http://localhost:5005/webhooks/rest/webhook
```

### Example JSON Payload:

```json
{
  "sender": "user123",
  "message": "What is EPR?"
}
```

### Example Response:

```json
[
  {
    "recipient_id": "user123",
    "text": "EPR stands for End Point Rate. It's a method used to calculate shoreline change rates..."
  }
]
```

---

## 📁 Project Structure

```bash
📁 actions/            # Custom action code in Python
📁 data/               # NLU data, stories, and rules
📁 models/             # Trained Rasa models
📄 domain.yml          # Intents, responses, actions, and slots
📄 config.yml          # NLP pipeline and policies
📄 endpoints.yml       # Endpoint configs (e.g., action server)
📄 credentials.yml     # API channels (e.g., REST, Socket.IO)
```

---

## 🧪 Helpful Rasa Commands

- Train model: `rasa train`
- Start chat: `rasa shell`
- Test NLU: `rasa shell nlu`
- Run API server: `rasa run --enable-api`
- Run actions: `rasa run actions`
- Visualize flows: `rasa visualize`

---

## 🧹 Clean Up

To delete old model files:

```bash
rasa delete model --model <model-name>
```

---

## 🙋‍♀️ Maintainer

Created and maintained by: **Your Name**  
GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 🚀 Happy Chatting!
