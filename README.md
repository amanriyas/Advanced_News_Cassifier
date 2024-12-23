Advanced News Classifier

Overview
This project implements an advanced news article classifier using GloVe (Global Vectors for Word Representation) embeddings and machine learning. Unlike traditional TF-IDF approaches, this classifier leverages word embeddings to capture contextual meanings and semantic relationships between words, resulting in more nuanced text classification.Features
- Text preprocessing including cleaning, lemmatization, and stop-word removal
- Document embedding generation using GloVe vectors
- Neural network-based classification of news articles
- Support for both training and testing data sets
- Flexible architecture supporting multiple news groups/categories

Technologies Used
- Java
- Stanford CoreNLP for text lemmatization
- Deeplearning4J for neural network implementation
- ND4J (N-Dimensional Arrays for Java) for array operations
- GloVe embeddings trained on Wikipedia 2014 + Gigaword 5

Project Structure
The project consists of several key components:

- `Glove.java`: Handles GloVe embeddings for individual words
- `NewsArticles.java`: Base class for news article management
- `HtmlParser.java`: Parses HTML news content
- `Toolkit.java`: Utilities for loading GloVe and news data
- `ArticlesEmbedding.java`: Generates document-level embeddings
- `AdvancedNewsClassifier.java`: Main classifier implementation

Prerequisites
- Java Development Kit (JDK 17 or above)
- IntelliJ IDEA (recommended IDE)
- Maven (for dependency management)

Setup and Installation
1. Clone the repository
2. Open the project in IntelliJ IDEA
3. Ensure all dependencies are properly loaded through Maven
4. Verify that the resources folder contains:
    - GloVe file (`glove.6B.50d_Reduced.csv`)
    - News articles in the News subfolder

    - Running the Classifier
1. Load the project in your IDE
2. Run the main method in the AdvancedNewsClassifier class
3. The classifier will:
    - Process and embed the training data
    - Train the neural network
    - Classify the test articles
    - Output the classification results

Output Format
The classifier outputs the news articles grouped by their predicted categories. Example output:
```
Group 1
[News articles in group 1...]

Group 2
[News articles in group 2...]
```

Notes
- The GloVe file used is a reduced version containing 38,534 unique words
- Each word is represented by a 50-dimensional vector
- The system automatically handles document length normalization for consistent neural network input

Performance Considerations
- Text preprocessing is optimized to run only once per document.
- Embedding generation is cached to improve performance
- Average execution times:
    - GloVe loading: < 280ms
    - News loading: < 30ms
    - Text preprocessing: < 13ms per document
    - Embedding generation: < 8ms per document
