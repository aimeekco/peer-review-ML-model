import openai 
from transformers import BertTokenizer, BertForSequenceClassification

openai.api_key = "api key"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def get_scores(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].tolist()
    return scores

def generate_suggestions(text, category, score):
    prompt = f"Read the following research paper text and provide constructive suggestions for improvement in the category '{category}', which has a score of {score} out of 10:\n\n{text}\n\n Suggestions:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    suggestions = response.choices[0].text.strip()
    return suggestions

# sample input 
paper_text = "sample long input text"

rubric_categories = [
    "Summary and Understanding",
    "Quotes and Citations",
    "Argumentative Coherence and Supporting Evidence",
    "Critical Analysis and Integration with Outside Sources and Originality",
    "Organization and Presentation",
    "Mechanics and Writing Quality",
    "Proper Spelling",
    "Formatting"
]

bert_scores = get_scores(paper_text)
details = ["Methodology", "Results", "Experiments", "Analysis"]

# generate suggestions based on BERT scores
suggestions_dict = {}
for category, score in zip(rubric_categories, bert_scores):
    suggestions = generate_suggestions(paper_text, category, score, details)
    suggestions_dict[category] = suggestions
    
final_output = {
    "scores": dict(zip(rubric_categories, bert_scores)),
    "overall_score": sum(bert_scores) / len(bert_scores),
    "suggestions": suggestions_dict
}

print(final_output)