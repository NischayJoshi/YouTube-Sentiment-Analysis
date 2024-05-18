import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stemmer, lemmatizer, and stop words
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and numbers
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", "", text)
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Remove stop words
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    words = filtered_words
    
    # Apply stemming
    stemmed_words = []
    for word in words:
        stemmed_word = ps.stem(word)
        stemmed_words.append(stemmed_word)
    
    # Apply lemmatization
    lemmatized_words = []
    for word in stemmed_words:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemmatized_word)
    
    # Join words back into a single string
    processed_text = " ".join(lemmatized_words)
    
    return processed_text

def main():
    # Ask user for input and output file names
    input_file = input("Enter the input CSV file name: ")
    output_file = input("Enter the output CSV file name: ")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Print out the column names for debugging purposes
    print("Columns in the CSV file:", df.columns)
    
    # Check if 'text' column exists in the dataframe
    if 'text' not in df.columns:
        print("The CSV file must contain a 'text' column.")
        return
    
    # Preprocess the text column
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Save the processed data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
