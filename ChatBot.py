import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure the 'punkt' tokenizer is downloaded
nltk.download("punkt")

# Define intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up", "Hello there", "Hey, How are you", "Anybody", "Hey there"],
        "responses": ["Hi there!", "Hello!", "Hey!", "I'm fine, thank you. How can I assist you today?", "Nothing much. How can I help you?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care", "bye bye", "Nice to chat with you", "See you later buddy"],
        "responses": ["Goodbye!", "See you later!", "Take care!", "Bye bye, thanks for reaching out.", "Have a nice day!", "See you later!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it", "Thanks for your quick response", "Thank you for providing the valuable information", "Awesome, thanks for helping"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!", "Happy to help you.", "Thanks for reaching out to me.", "It's my pleasure to help you."]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot created to assist you with various queries.", "My purpose is to help you with any questions you might have.", "I can answer questions and provide assistance on various topics."]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do", "What help can you offer?", "How can you assist me?"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?", "I can help with a variety of tasks. What do you need?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information. You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.", "Improving your credit score involves paying bills on time, reducing debt, and checking your credit report for errors."]
    },
    {
        "tag": "no_answer",
        "patterns": [],
        "responses": ["I'm sorry, I didn't understand that. Could you please rephrase?", "Can you provide more details?", "I'm not sure I understand. Can you elaborate?"]
    }
]

# Initialize the vectorizer and the classifier
vectorizer = TfidfVectorizer()
classifier = LogisticRegression(random_state=0, max_iter=1000)

# Prepare training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Transform patterns into TF-IDF features and fit the classifier
x = vectorizer.fit_transform(patterns)
y = tags
classifier.fit(x, y)

# Define the chatbot response function
def chatbot_response(text):
    input_text = vectorizer.transform([text])
    predicted_tag = classifier.predict(input_text)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return response

# Interactive chatbot session
print("Chatbot: Hi there! I'm here to help you. You can type 'exit' anytime to end the chat.")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
