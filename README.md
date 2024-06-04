# Chatbot Assistant

This is an AI-based chatbot assistant built using Python, TF-IDF vectorization, and logistic regression. It can handle various intents such as greetings, farewells, expressing thanks, and answering questions about specific topics like budgeting and credit scores.

## Features

- **Greeting and Farewell:** The bot can recognize and respond to greetings and farewells.
- **Thankfulness:** The bot can handle expressions of thanks.
- **Help and Assistance:** The bot can provide assistance on various topics, including budgeting and credit scores.
- **Error Handling:** Enhanced error handling to manage unknown inputs gracefully.
- **Interactive Session:** Begins with a friendly greeting and offers a clear exit option for users to end the conversation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chatbot-assistant.git
    cd chatbot-assistant
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the NLTK 'punkt' tokenizer:
    ```python
    import nltk
    nltk.download("punkt")
    ```

## Usage

1. Run the chatbot:
    ```bash
    python chatbot.py
    ```

2. Interact with the chatbot:
    - Type your messages in the console.
    - Type "exit" to end the chat.

## Example Interaction

```bash
Chatbot: Hi there! I'm here to help you. You can type 'exit' anytime to end the chat.
User: Hi
Chatbot: Hello!
User: How can you help me with budgeting?
Chatbot: To make a budget, start by tracking your income and expenses...
User: Thank you
Chatbot: You're welcome!
User: exit
Chatbot: Goodbye! Have a great day!
