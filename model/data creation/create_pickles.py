import os
import pickle

commonFilePath = "/Users/aimeeco/peer-review-ML-model/commonfile.txt"
rel_path = '/Users/aimeeco/peer-review-ML-model/data/Annotated_reviews@1208/' 
pickle_dir = "/Users/aimeeco/peer-review-ML-model/data"

# store data in lists
tags = []
fileId = []
sentences = []
sentencesOk = []
fileIdOk = []
tagsOk = []

# open common file to write
commonFile = open(commonFilePath, "w+")
commonLines = []

# processes one file
def processFile(file, fileName):
    lines = file.read().strip().split(']]')
    lines = lines[:-1]
    num_row = len(lines)
    for i in range(num_row):
        lines[i] = lines[i] + ']]'
        divided = lines[i].split('[[')
        if len(divided) == 2:
            divided[1] = '[[' + divided[1]
            tags.append(divided[1])
            fileId.append(fileName)
            sentences.append(divided[0])

# reads one file
def readFile(fileName):
    file_path = os.path.join(rel_path, fileName)
    with open(file_path, encoding='utf-8') as singleFile:
        processFile(singleFile, fileName)

# read all files in directory
def readFiles(path):
    for file in os.listdir(path):
        if not file.startswith('.'):
            readFile(file)
    commonFile.close()

# read all files in path
readFiles(rel_path)

num_rows = len(sentences)
print(f"Number of sentences read: {num_rows}")

# separate tags and process sentences
def separate_tags():
    for i in range(num_rows):
        tag = tags[i].replace(' ', '').upper()[1:-1]
        sep_tag = tag.split('],[')
        if len(sep_tag) == 4:
            sep_tag[0] = sep_tag[0].replace('[', '')
            sep_tag[3] = sep_tag[3].replace(']', '')
            sentencesOk.append(sentences[i])
            fileIdOk.append(fileId[i])
            tagsOk.append(sep_tag)

separate_tags()

# debugging
print(f"Number of processed sentences: {len(sentencesOk)}")
print(f"Number of processed file IDs: {len(fileIdOk)}")
print(f"Number of processed tags: {len(tagsOk)}")

# save to pickle files
with open(os.path.join(pickle_dir, "sentence.pickle"), "wb") as pickle_out:
    pickle.dump(sentencesOk, pickle_out)

with open(os.path.join(pickle_dir, "fileid.pickle"), "wb") as pickle_out:
    pickle.dump(fileIdOk, pickle_out)

with open(os.path.join(pickle_dir, "tags.pickle"), "wb") as pickle_out:
    pickle.dump(tagsOk, pickle_out)

print("Pickle files created successfully.")