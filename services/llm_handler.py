import os
from openai import OpenAI
from typing import List, Tuple

class LLMHandler:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.model = "gpt-4-turbo-preview"
        # Initialize the OpenAI client with your API key
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def find_best_match(self, query: str, options: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
        """
        Craft a prompt to send to the LLM and interpret its response to find the best match.
        """
        prompt = f"Given the query: '{query}', which of the following text options includes the best match?\n\n"
        for i, (_, _, text) in enumerate(options):
            prompt += f"Option {i}: {text}\n\n"

        prompt += "Select the best option number and output only the option:"

        # Create chat completion
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )

        # Parsing the response to extract the option chosen by the LLM
        try:
            # The LLM response is expected to indicate a preference. Here, we simulate extracting the preferred option number.
            # Extract the response text
            response_text = chat_completion.choices[0].message.content
            # Extracting the first digit as the chosen option, assuming the LLM responds with something like "Option 1 is the best match."
            best_option_number = int(''.join(filter(str.isdigit, response_text)))
            best_match = options[best_option_number]
        except (IndexError, ValueError, KeyError) as e:
            print(f"Error processing the LLM response: {e}")
            best_match = None

        return best_match
