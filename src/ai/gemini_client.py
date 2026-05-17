import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


def get_model(temperature: float = 0.0):
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(temperature=temperature),
    )


async def generate(prompt: str, temperature: float = 0.0) -> str:
    model = get_model(temperature)
    response = await model.generate_content_async(prompt)
    return response.text
