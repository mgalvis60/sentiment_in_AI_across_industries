## Sentiment in AI across industries ##
Understand people's sentiment regarding AI across several industries and tools, based on news articles.

## Main Goal ##
This project collection ~200K news articles on data science, machine learning, and artificial intelligence using webscraping since 2022 to 2026. Using NLP techniques the two main objects are:
1) Identify industries and their companies that are most likely to be impacted by AI over the next several years.
2) Determine how those industries and their companies will be impacted (positively, negatively, hard-to-say, etc.) and by what means / technologies (e.g., automation, augmentation, workflow redesign, cost reduction, etc.).
   
## Get and clean data ##
•	Read the dataset of artificial intelligence using webscraping and applied initial EDA in the title and text columns. \
•	Clean title to drop source to avoid bias in the results. \
•	Clean text column removed in information before the title and after the main content article, to avoid considering menu options, publicity, content related to other articles, sources, and videos. \
•	Remove articles with fewer than 50 words, focus on video and image generation. \
•	Develop a references sentence for filter articles not related to impact of AI across industries using a transformer model called all-MiniLM-L6-v2 to compute cosine similarity, and filter documents using a threshold.
## Topic detection ##
•	A classic method, Latent Dirichlet Allocation (LDA), is select to discover the hidden topics in the articles. \
•	Apply a lemmatization process to a better comparison with lowercase tokens, removing stop words, and deleting punctuation. \
•	Create bigrams and trigrams to find more common concepts. \
•	Employ a grid search to establish the adequate number of topics, in this case was 10 topics. \
•	Name topics and analysis frequency across the corpus. \

## Entity extraction ##
•	Extract organizations and technologies of the text using a (NER) Bert-base-cased pretrain model of Hugging Face called ner-bert-base-cased-ontonotesv5-englishv4. This model was fine-tuning using different types of entities, such as organization, product, event, and person on several English datasets around weblogs, pr newsier, newsgroups, among others. \
•	Great visual plots of the main companies and technologies related to AI in the corpus.

## Sentimental analysis ##
•	In this case the approach is to train a model, rather than taking an algorithm already designed for this specific use case. \
•	A basic Bert model (distilbert-base-uncased) is use to be fine-tuned on a sentiment dataset. \
•	The dataset selected to execute the tunning process is community-datasets/per_sent, which captures the sentiment of a writer in a news article. It contains 5.3k documents with the possible labels (Positive, Neutral, Negative).

## Sentimental application ##
•	Apply the model to the clean corpus of AI across several industries and tools to understand the perception of people regarding this new technology. \
•	Run a sentiment analysis across the 10 topics identity in topic modeling section. \
•	Execute a sentiment analysis across entities focus on the top organizations and technologies. \
•	Check the average evolution over 2022 to 2026 for topic modeling, organizations, and technologies. \
If you want to make a deep dive into the main takeaways, feel free to explore the insights folder with the executive presentation and if you desire to explore the technical results enter the notebooks folder.
