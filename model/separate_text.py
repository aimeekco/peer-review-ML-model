import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() 
    return text

def separate_sections(text):
    sections = ["Introduction", "Methodology", "Results", "Discussion", "Conclusion"]
    section_texts = {section: "" for section in sections}
    
    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        if any(section in line for section in sections):
            for section in sections:
                if section in line:
                    current_section = section
                    break
        if current_section:
            section_texts[current_section] += line + "\n"
    return section_texts

pdf_path = "../data/sample_papers/morrison-et-al-2021.pdf"
text= extract_text_from_pdf(pdf_path)
section_texts = separate_sections(text)

for section, content in section_texts.items():
    print(f"Section: {section}\nContent: {content[:500]}\n")
    print("="*80)