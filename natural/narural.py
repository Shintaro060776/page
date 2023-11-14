import csv
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

jokes = []
with open('shortjokes.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        jokes.append(row[1])

tokenized_jokes = [word_tokenize(joke) for joke in jokes]

with open('tokenized_jokes.json', 'w', encoding='utf-8') as file:
    json.dump(tokenized_jokes, file, ensure_ascii=False, indent=4)

with open('tokenized_jokes.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tokenized Joke'])
    for tokens in tokenized_jokes:
        writer.writerow([' '.join(tokens)])
