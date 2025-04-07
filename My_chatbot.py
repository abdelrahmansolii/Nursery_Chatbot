import os
import google.generativeai as genai
import random
from typing import Dict, List


class NurseryChatbot:
    """CLI chatbot with nursery Q&A + optional stories!."""

    def __init__(self):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY", "your-api-key-here"))
        except Exception as e:
            print(f"Error configuring API: {str(e)}")
            raise

        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        self.system_instruction = """
        You are Sunny from Sunshine Nursery. Primary focus: nursery information.
        Only provide stories when explicitly requested with words like 'story' or 'tale'.
        """

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            system_instruction=self.system_instruction
        )

        self.conversation_history = []
        self.max_history = 10

        # Sample stories
        self.stories = [
            {
                "title": "The Honest Rabbit",
                "content": "Lily the rabbit found a lost necklace and returned it...",
                "age": "4+",
                "moral": "Honesty brings happiness"
            }
        ]

    def generate_response(self, user_input: str) -> str:
        try:
            lower_input = user_input.lower()

            # Handle special cases
            if any(word in lower_input for word in ['bye', 'exit']):
                return "Thank you for chatting with Sunshine Nursery! 🌻"

            if any(word in lower_input for word in ['story', 'tale']):
                return self._provide_story(user_input)

            # Normal Q&A
            context = "\n".join(
                f"User: {msg['user']}\nAssistant: {msg['bot']}"
                for msg in self.conversation_history
            )
            prompt = f"{context}\nUser: {user_input}\nAssistant:"

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            self._update_conversation(user_input, response_text)
            return response_text

        except Exception as e:
            return f"I'm having trouble responding. Error: {str(e)}"

    def _provide_story(self, user_input: str) -> str:
        """Provides a story without disrupting main Q&A flow"""
        age = next((int(word) for word in user_input.split() if word.isdigit()), None)
        available = [s for s in self.stories if not age or int(s['age'][0]) <= age]

        if not available:
            return "No stories available. Ask about our nursery programs!"

        story = random.choice(available)
        self._update_conversation(user_input, f"Told story: {story['title']}")

        return (
            f"📖 {story['title']} (Age {story['age']})\n\n"
            f"{story['content']}\n\n"
            f"✨ Moral: {story['moral']}\n\n"
            f"Ask about nursery or say 'another story'!"
        )

    def _update_conversation(self, user_input: str, response: str) -> None:
        self.conversation_history.append({"user": user_input, "bot": response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)


def main():
    bot = NurseryChatbot()
    print("\n🌻 Welcome to Sunshine Nursery Chatbot")
    print("🌱 Ask about nursery or request a story. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! 👋")
                break

            response = bot.generate_response(user_input)
            print("\nSunny:", response, "\n")

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()