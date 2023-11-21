import MeCab
import re

mecab = MeCab.Tagger("-Owakati")


def clean_lyrics(lyrics):
    lyrics = re.sub(r'<[^>]+>', '', lyrics)
    lyrics = re.sub(r'[^\w\s]', '', lyrics)
    lyrics = lyrics.replace('\n', ' ')
    return lyrics


def tokenize_japanese(lyrics):
    return mecab.parse(lyrics).strip()


def preprocess_lyrics(file_path, output_path):
    with open(file_path, 'r', encoding='shift_jis') as file:
        lyrics = file.read()
        cleaned_lyrics = clean_lyrics(lyrics)
        tokenized_lyrics = tokenize_japanese(cleaned_lyrics)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(tokenized_lyrics)


input_file_path = 'line.txt'
output_file_path = 'preprocessed_lyrics.txt'

preprocess_lyrics(input_file_path, output_file_path)
