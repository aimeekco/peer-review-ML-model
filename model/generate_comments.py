import openai 
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_comment(section):
    prompt = f"""
    You are a peer reviewer for a scientific paper. The paper is titled "{paper_details['title']}" and is authored by {paper_details['authors']}. Your task is to provide detailed and constructive feedback on the {section} section of the paper. Please ensure your comments are professional, thorough, and helpful for the authors to improve their work.

    Section: {section}
    """
    
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature= 0.7
    )
    
    return response.choices[0].text.strip()

#example usage
paper_details = {
    "title": "Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015",
    "authors": "L. V. Morrison, F. R. Stephenson, C. Y. Hohenkerk, M. Zawilski"
}

sections = ["Introduction", "Methodology", "Results", "Discussion", "Conclusion"]

for section in sections:
    comment = generate_comment(section, paper_details)
    print(f"Section: {section}\nComment: {comment}\n")