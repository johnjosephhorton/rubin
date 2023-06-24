import openai
from dotenv import load_dotenv
import os
import sys


class LanguageModel:
    def __init__(self, model, family, temperature):
        load_dotenv()  # Load environment variables from .env file
        self.model = model
        self.family = family
        self.temperature = temperature
        if family == "openai":
            if model == "text-davinci-003":
                self.call_llm = self.call_openai_api
            elif model == "gpt-3.5-turbo" or  model == "gpt-4":
                self.call_llm = self.call_open_ai_apt_35
            else:
                raise Exception("Model not supported")
        else:
            raise Exception("Family not supported")

    def call_open_ai_apt_35(self, prompt):
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": ""},
            {"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"]

    def call_openai_api(self, prompt):
        openai.api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable

        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=self.temperature
        )

        return response.choices[0].text.strip()