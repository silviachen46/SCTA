from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from typing import List

CODE_INDUCER = """
Return ```python your_code_here ``` with NO other texts. your_code_here is a placeholder.
your code:

"""

# Azure OpenAI client setup
class OpenAiClient:
    def __init__(self, client_type = "Azure"):
        if client_type == "GPT":
            self.client = OpenAI(api_key="api-key")
        else:
            self.client = AzureOpenAI(
                api_key="api_key_here",
                api_version="2024-08-01-preview",
                azure_endpoint="endpoint_url"
            )

    def call_openai_gpt(self, prompt, sys_prompt=None):
        if sys_prompt is None:
            sys_prompt = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            # Attempt to get a response from the API
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Adjust the model name as needed
                messages=messages
            )

            # Extract and print the response content
            response_content = response.choices[0].message.content
            print(response_content)
            return response_content
        except Exception as e:
            # Print an error message if something goes wrong
            print("An error occurred while fetching the response:", e)

        return response.choices[0].message.content
    
    def gpt_parsed_call(self, prompt, format):
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a reviewer, give review as instructed by the user."},
                {"role": "user", "content": prompt},
            ],
            response_format=format,
        )

        result = completion.choices[0].message.parsed.model_dump() # return a dictionary
        return result
        
    
    

class GeneralReview(BaseModel):
    decision: str
    reason: str

class Selection(BaseModel):
    selected: str


class GeneSelection(BaseModel):
    selected: List[str]
    reasoning: str

    
    
