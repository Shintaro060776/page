import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

tokenized_jokes = []
with open('tokenized_jokes.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        tokenized_jokes.append(row[0].split())

normalized_jokes = []
for tokens in tokenized_jokes:
    normalized_tokens = [stemmer.stem(
        word.lower()) for word in tokens if word.lower() not in stop_words]
    normalized_jokes.append(normalized_tokens)

with open('normalized_jokes.json', 'w', encoding='utf-8') as file:
    json.dump(normalized_jokes, file, ensure_ascii=False, indent=4)

with open('normalized_jokes.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Normalized Joke'])
    for tokens in normalized_jokes:
        writer.writerow([' '.join(tokens)])
