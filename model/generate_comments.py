from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_comment(section, paper_details):
    prompt = f"""
    You are a peer reviewer for a scientific paper. The paper is titled "{paper_details['title']}" and is authored by {paper_details['authors']}. Your task is to provide detailed and constructive feedback on the {section} section of the paper. Please ensure your comments are professional, thorough, and helpful for the authors to improve their work.

    Section: {section}
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    n=1,
    temperature=0.7)

    return response.choices[0].message.content.strip()

# example usage after running separate_text.py
paper_details = {
    "title": "Addendum 2020 to Measurement of the Earth's rotation: 720 BC to AD 2015",
    "authors": "L. V. Morrison, F. R. Stephenson, C. Y. Hohenkerk, M. Zawilski"
}

sections = ["Introduction: In a paper by three of the current authors—Stephensonet al. [1], herein referred to as paper 2016—a comprehensive compilation was presented of historical reports of solar and lunar eclipses in the period 720 BC to AD 2015, from which we deduced changes in the Earth’srate of rotation. That investigation indicated that there are changes in the rate on centennial as well as decadal time scales.", 
            "Methodology", 
            "Results", 
            "Discussion", 
            "Conclusion: The results for the changes in the Earth’s spin on a centennial and longer time scale are displayed in table 4 . The actual measured ﬂuctuations in rate are plotted in ﬁgure 4 and can be obtained from electronic supplementary material, table S6. The new and revised data in this addendum to our 2016 paper [ 1] have allowed us to modify and improve on the reliability of the results in that paper. The observed deceleration in the Earth’s deceleration where t=(year−1750)/1400 spin is signiﬁcantly less than that predicted by tidal friction. The densiﬁcation of our dataset in the period AD 800–1600 supports our conclusion in the 2016 paper that there is a ﬂuctuation inthe rate of spin on a centennial time scale. However, there are not enough reports of critical eclipse observations before 136 BC to determine whether this ﬂuctuation is truly part of a periodic term of about 14 centuries."
            ]

for section in sections:
    comment = generate_comment(section, paper_details)
    print(f"Section: {section}\nComment: {comment}\n")
