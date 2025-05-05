import nltk

def download_nltk_resources():
    print("Downloading NLTK resources...")
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')
    print("Downloading wordnet...")
    nltk.download('wordnet')
    print("Downloading omw-1.4...")
    nltk.download('omw-1.4')
    print("Download complete.")

if __name__ == "__main__":
    download_nltk_resources()