from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, List
import logging
from functools import wraps
import urllib.parse
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise


class NurseryChatbot:
    """Enhanced chatbot with nursery Q&A + story features.."""

    def __init__(self):
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }

        self.system_instruction = """
        You are Sunny, the assistant for Sunshine International Nursery (est. 2010).
        Your PRIMARY role is nursery information. SECONDARY role is providing age-appropriate stories when explicitly requested.

        CORE NURSERY KNOWLEDGE:
        - Location: 56 Toman Bai St, Nasr City
        - Hours: 8AM-1PM (extended to 3PM)
        - Curriculum: British EYFS + Arabic/French
        - Fees: 10K L.E/semester (10% sibling discount)
        - Ages: 6mo-5yrs

        STORYTELLING RULES:
        - Only provide stories when explicitly asked
        - Stories must be 100-300 words, with clear morals
        - Tag with age appropriateness (3+, 4+, 5+)
        """

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            system_instruction=self.system_instruction
        )

        self.conversation_history = []
        self.max_history = 10
        self.story_mode = False

        # Pre-defined simple responses
        self.common_qa = {
            "goodbye": "Thank you for contacting Sunshine Nursery! 🌻 Have a wonderful day!",
            "map": self._generate_map_response(),
            "math": "Here's the answer: "
        }

        # Sample curated stories (could be expanded)
        self.stories = [
            {
                "title": "The Sharing Bear",
                "content": "Benny the bear never shared his honey. One day, his friends stopped playing with him...",
                "age": "3+",
                "moral": "Sharing makes friendships stronger",
                "keywords": ["share", "kindness"]
            },
            {
                "title": "The Curious Kitten",
                "content": "Milo the kitten explored too far and got lost. He remembered his mother's advice about...",
                "age": "4+",
                "moral": "Curiosity should be balanced with caution",
                "keywords": ["explore", "safety"]
            }
        ]

    def _generate_map_response(self) -> str:
        address = "56 Toman Bai Street, Nasr City, Cairo"
        map_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(address)}"
        return f"We're at {address}. Map: {map_link}"

    def generate_response(self, user_input: str) -> str:
        try:
            lower_input = user_input.lower()

            # Handle special cases
            if any(word in lower_input for word in ['bye', 'goodbye']):
                return self.common_qa["goodbye"]

            if any(word in lower_input for word in ['map', 'location']):
                return self.common_qa["map"]

            if any(word in lower_input for word in ['math', 'calculate']):
                return self._handle_math_query(user_input)

            # Story handling
            if any(word in lower_input for word in ['story', 'tale', 'bedtime']):
                return self._handle_story_request(user_input)

            # Normal nursery Q&A
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
            logger.error(f"Error: {str(e)}")
            return "Let's try something else! Ask about our nursery programs."

    def _handle_story_request(self, user_input: str) -> str:
        """Handles story requests while maintaining nursery context"""
        age = next((int(word) for word in user_input.split() if word.isdigit()), None)
        matching_stories = [
            s for s in self.stories
            if not age or int(s['age'][0]) <= age
        ]

        if not matching_stories:
            return "No suitable stories found. Try asking about our nursery programs!"

        story = random.choice(matching_stories)
        self._update_conversation(user_input, f"Shared story: {story['title']}")

        return (
            f"📖 {story['title']} (Age {story['age']})\n\n"
            f"{story['content']}\n\n"
            f"✨ Moral: {story['moral']}\n\n"
            f"Want another story or ask about nursery?"
        )

    def _handle_math_query(self, query: str) -> str:
        if '*' in query or 'x' in query:
            nums = [int(s) for s in query.split() if s.isdigit()]
            if len(nums) == 2:
                return f"{self.common_qa['math']}{nums[0]} × {nums[1]} = {nums[0] * nums[1]}"
        return f"{self.common_qa['math']}Try: 'What's 3 times 4?'"

    def _update_conversation(self, user_input: str, response: str) -> None:
        self.conversation_history.append({"user": user_input, "bot": response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)


# Initialize chatbot
chatbot = NurseryChatbot()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({'status': 'error', 'message': 'Empty input'}), 400

    try:
        response = chatbot.generate_response(user_input)
        return jsonify({
            'status': 'success',
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Try again later'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)