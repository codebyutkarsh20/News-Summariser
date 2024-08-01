import openai
from dotenv import load_dotenv
import requests
import os
import streamlit as st
import json
import time

load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')
news_api_key = os.getenv('NEWS_API_KEY')
openai.api_key = openai_api_key

model = "gpt-4"

def get_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            final_news = []

            for article in articles:
                title = article.get("title", 'No Title')
                author = article.get("author", 'Unknown Author')
                source_name = article["source"].get('name', 'Unknown Source')
                description = article.get("description", 'No Description')
                url = article.get("url", 'No URL')

                title_description = f"""
                Title: {title},
                Author: {author},
                Source: {source_name},
                Description: {description},
                URL: {url}
                """
                final_news.append(title_description.strip())

            return final_news

        else:
            print(f"Error: Unable to fetch news (Status Code: {response.status_code})")
            return []

    except requests.exceptions.RequestException as e:
        print("Error occurred during API Request", e)
        return []

class AssistantManager:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None
        self.assistant_id = None
        self.thread_id = None

    def create_assistant(self, name, instructions, tools):
        if not self.assistant:
            self.assistant = openai.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model=self.model
            )
            self.assistant_id = self.assistant.id

    def create_thread(self):
        if not self.thread:
            self.thread = openai.beta.threads.create()
            self.thread_id = self.thread.id

    def add_msg_to_thread(self, role, content):
        if self.thread:
            openai.beta.threads.messages.create(
                thread_id=self.thread_id,
                role=role,
                content=content
            )

    def run_assistant(self, instructions):
        if self.thread and self.assistant:
            self.run = openai.beta.threads.runs.create_and_poll(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                instructions=instructions
            )

    def process_message(self):
        if self.thread:
            messages = openai.beta.threads.messages.list(thread_id=self.thread_id)
            summary=[]
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)
            self.summary="\n".join(summary)
            
            print(f"Summary---->{role.capitalize()}:--->{response}")

    def call_required_function(self, required_action):
        tools_outputs = []
        for action in required_action.get("tool_calls", []):
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if func_name == "get_news":
                output = get_news(topic=arguments["topic"])
                tools_outputs.append({"tool_call_id": action["id"], "output": "\n".join(output)})

        print("Submitting output back to the assistant...")
        openai.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread_id,
            run_id=self.run.id,
            tool_outputs=tools_outputs
        )

    def get_summary(self):
        return self.summary

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(2)
                run_status = openai.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=self.run.id
                )
                print(f"Run status: {run_status}")
                if run_status.status == "completed":
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    print("Function calling now...")
                    self.call_required_function(run_status.required_action.submit_tool_outputs.model_dump())

def main():
    manager = AssistantManager()

    st.title("News Summarizer")
    with st.form(key="user_input_form"):
        instructions = st.text_input("Enter topic")
        submit = st.form_submit_button(label="Run Assistant")

        if submit:
            manager.create_assistant(
                name="News Summarizer",
                instructions="You are a personal article summarizer assistant.",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "description": "Get the list of articles/news for the given topic",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic for the news, e.g. bitcoin",
                                    }
                                },
                                "required": ["topic"],
                            },
                        },
                    }
                ]
            )
            manager.create_thread()

            manager.add_msg_to_thread(
                role="user",
                content=f"Summarize the news on this topic: {instructions}?"
            )
            manager.run_assistant(instructions="Summarize the news")

            manager.wait_for_completion()
            summary = manager.get_summary()
            st.write(summary)

if __name__ == "__main__":
    main()
