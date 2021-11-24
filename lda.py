from gensim.models import LdaModel
from gensim.corpora import Dictionary, dictionary
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


class LDA:
    def __init__(self, num_features):
        self.num_features = num_features
        self.model = None
        self.dictionary = None

    def fit(self, docs):
        self.model, _, self.dictionary = train_lda_model(docs, num_topics=self.num_features)

    def transform(self, docs):
        if self.model == None or self.dictionary == None:
            raise ValueError('transform called before fit')
        tokenized_docs = tokenize(docs)
        test_bow = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        features, _ = self.model.inference(test_bow)
        return features

def train_lda_model(docs, num_topics=100):
    docs = tokenize(docs)

    # Remove rare and common tokens.

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Train LDA model.

    # Set training parameters.
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary. Without this line id2token is not initialized
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    return model, corpus, dictionary


def tokenize(docs):
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Compute bigrams.

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    return docs
