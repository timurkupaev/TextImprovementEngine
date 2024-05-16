# Text Improvement Engine

## Objective
This command line tool analyzes a given text and suggests improvements based on the similarity to a list of standardized phrases.


## Technologies Used

- **Python**
- **spaCy** 
- **BERT**
- **Transformers**
- **NLTK**: 
- **sklearn**
- **Matplotlib**
- **Seaborn**


## Design Rationale and Conclusions 

For **detailed experimentation, development process, and further improvements**, please refer to one of the following files: 
-  `notebooks/development_process.ipynb` 
- `notebooks/development_process.pdf`

For this task, two approaches were considered: spaCy and BERT.

**Approach 1**: The spaCy approach uses the spaCy library to extract phrases from a sample text and match them with predefined standardized phrases based on similarity measures. The sample text is processed to identify key phrases involving verbs and other Parts of Speech (POS) and filtered for irrelevant words (stopwords, using NLTK).

**Approach 2**: The BERT approach breaks the sample text into sentences and further splits each sentence into phrases, calculates similarity scores against standardized phrases using a BERT model, and identifies the closest matches. The BERT approach was enhanced by adding spaCy phrases.

For submission, **Approach 1** was chosen. While the BERT + spaCy approach shows promising results with scores over 0.8, the contextual relevance of the suggestions is higher for the spaCy approach alone. Further improvement of the BERT approach is promising and could potentially outperform the spaCy approach.



## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/timurkupaev/TextImprovementEngine.git
    cd TextImprovementEngine
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the NLTK stopwords and  spaCy model:
	```bash 
	python -m nltk.downloader stopwords
	```
	
    ```bash
    python -m spacy download en_core_web_lg
    ```

## Usage
1. Run the tool with default values:
    ```bash
    python src/text_improvement_engine.py
    ```
    This will use:
    - Input text file: `data/sample_text.txt`
    - Standardized phrases file: `data/Standardised terms.csv`
    - Similarity threshold: `0.6`

2. Run the tool with custom input files and threshold:
    ```bash
    python src/text_improvement_engine --input "custom_path/input.txt" --standardized "custom_path/Standardised_terms.csv" --threshold 0.75
    ```
    This allows you to specify:
    - `--input`: Path to your input text file
    - `--standardized`: Path to the standardized phrases CSV file
    - `--threshold`: Similarity threshold (default is 0.6)
	
	Note: Regardless of the threshold value for each original phrase only the top three recommended replacements will be shown. 

## Output

The output provides a list of suggestions to replace phrases in the input text with their more "standard" versions. Each suggestion shows the original phrase, the three recommended replacements, and their similarity scores. The suggestions are sorted by similarity scores in descending order.

The tool will output suggestions with similarity scores in the following format:
Original Phrase: [Original phrase from input text]
Suggestion: [Suggested standardized phrase] - Score: [Similarity score]
Suggestion: [Suggested standardized phrase] - Score: [Similarity score]
Suggestion: [Suggested standardized phrase] - Score: [Similarity score]

