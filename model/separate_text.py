import pdfplumber
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to detect section titles using GPT-4
def detect_section_titles(text):

    prompt = f"Extract section titles from the following text:\n\n{text}"  

    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an assistant that extracts section titles from academic papers."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    n=1,
    stop=None,
    temperature=0.7)

    section_titles = response.choices[0].message.content.strip().split('\n')
    section_titles = [title.strip() for title in section_titles if title.strip()]
    return section_titles

# Function to separate sections based on detected titles
def separate_sections(text, section_titles):
    section_patterns = [re.compile(rf'\b{re.escape(title)}\b', re.IGNORECASE) for title in section_titles]
    section_texts = {title: "" for title in section_titles}

    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        for i, pattern in enumerate(section_patterns):
            if pattern.search(line):
                current_section = section_titles[i]
                break
        if current_section:
            section_texts[current_section] += line + "\n"
    return section_texts

# Example usage
pdf_path = "../data/sample_papers/co-et-al-2023.pdf"

# Step 1: Extract text
text = extract_text_from_pdf(pdf_path)
print("Extracted Text:")
print(text[:1000])  # Print first 1000 characters to verify extraction
print("="*80)

# Step 2: Detect section titles using GPT-4
section_titles = detect_section_titles(text)
print("Detected Section Titles:")
print(section_titles)
print("="*80)

# Step 3: Separate sections
section_texts = separate_sections(text, section_titles)

# Print the first part of the content for each section
for section, content in section_texts.items():
    print(f"Section: {section}\nContent: {content[:1000]}\n")
    print("="*80)
