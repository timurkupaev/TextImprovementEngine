import spacy
from nltk.corpus import stopwords
import csv
import argparse
import nltk
import logging

logging.getLogger('nltk').setLevel(logging.CRITICAL)

class PhraseExtractor:

    def __init__(self, model_name="en_core_web_lg"):
        """
        Initialize the PhraseExtractor with  Spacy model, download NLTK stopwords if user hasn't download it 
        :param model_name: The name of the spaCy model to load.
        """
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = self.load_spacy_model(model_name)

    def download_nltk_data(self):
        # download NLTK stopwords 
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords', quiet=True)

    def load_spacy_model(self, model_name):        
        # Load Spacy model
        try:
            return spacy.load(model_name)
        except OSError:
            raise OSError(f"The spacy model '{model_name}' is not downloaded. Please run:\n"
                          f"python -m spacy download {model_name}\n")

    def load_sample_text(self, file_path):
        # Load sample text from file
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return ""
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return ""

    def load_standardized_phrases(self, file_path):

        # Load phrases from csv file

        standardized_phrases = []
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    standardized_phrases.append(row[0])
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
        return standardized_phrases

    def extract_phrases(self, doc, min_words=3):
        """
        Extract phrases containing verbs from a Spacy document.

        :param doc: The Spacy document to process.
        :param min_words: Minimum number of words in a phrase.
        :return: A list of extracted phrases.
        """
        phrases = []
        for token in doc:
            if token.pos_ == "VERB" and token.text.lower() not in self.stop_words:
                phrase = doc[token.left_edge.i:token.right_edge.i + 1]
                if len(phrase) >= min_words and any(word.text.lower() not in self.stop_words for word in phrase):
                    phrases.append(phrase)
        return list(set(phrases))

    def generate_suggestions(self, filtered_phrases, standardized_docs, threshold):
        """
        Generate suggestions based on the similarity between extracted and standardized phrases.

        :param filtered_phrases: List of extracted phrases.
        :param standardized_docs: List of Spacy documents of standardized phrases.
        :param threshold: Similarity threshold for generating suggestions.
        :return: A list of suggestions with similarity scores.
        """
        suggestions = []
        for phrase in filtered_phrases:
            for std_doc in standardized_docs:
                cosine_similarity = phrase.similarity(std_doc)
                if cosine_similarity > threshold:
                    suggestions.append({
                        'original_phrase': phrase.text.strip(),
                        'suggested_standard_phrase': std_doc.text,
                        'similarity_score': cosine_similarity
                    })
        return suggestions

    def print_suggestions(self, suggestions):
        # Print the suggestions sorted by similarity score.

        suggestions = sorted(suggestions, key=lambda x: x['similarity_score'], reverse=True)
        phrase_suggestions = {}
        for suggestion in suggestions:
            phrase_suggestions.setdefault(suggestion['original_phrase'], []).append(suggestion)
        for phrase, sug_list in phrase_suggestions.items():
            print(f"\nOriginal Phrase: {phrase}")
            for suggestion in sug_list[:3]:
                print(f"  Suggestion: {suggestion['suggested_standard_phrase']} - Score: {suggestion['similarity_score']:.2f}")

    def process(self, input_path, standardized_phrases_path, threshold):
        """
        Process the input text to extract phrases and generate suggestions.

        :param input_path: Path to the input text file.
        :param standardized_phrases_path: Path to the standardized phrases CSV file.
        :param threshold: Similarity threshold for generating suggestions.
        """
        sample_text = self.load_sample_text(input_path)
        if not sample_text:
            return
        standardized_phrases = self.load_standardized_phrases(standardized_phrases_path)
        if not standardized_phrases:
            return
        # generating embeddings
        standardized_docs = [self.nlp(text) for text in standardized_phrases]
        # converting the sample text into a Spacy Doc object
        doc = self.nlp(sample_text)
        filtered_phrases = self.extract_phrases(doc)
        suggestions = self.generate_suggestions(filtered_phrases, standardized_docs, threshold)
        self.print_suggestions(suggestions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="data/sample_text.txt", help='Path to the sample text file')
    parser.add_argument('--standardized', type=str, default="data/Standardised terms.csv", help='Path to the standardized phrases CSV file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for similarity score')
    args = parser.parse_args()

    phrase_extractor = PhraseExtractor()
    phrase_extractor.process(args.input, args.standardized, args.threshold)
