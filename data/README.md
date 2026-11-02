# Data

Nothing in this folder is committed except this file (see `.gitignore`). Local runs of the
notebooks read and write their intermediate files here; in Google Colab they use a Google Drive
folder instead (set by `DATA_DIR` in the first code cell of each notebook).

## Raw input (not included)

The pipeline starts from one file you need to provide:

| File | Description |
|---|---|
| `df_news_AI.parquet` | About 200K English news articles on AI / machine learning / data science, published 2022 to February 2026, collected with web scraping |

Expected columns:

| Column | Type | Description |
|---|---|---|
| `url` | string | Address of the article |
| `date` | datetime | Publication date |
| `language` | string | Language code (all `en`) |
| `title` | string | Article title, usually with the source name appended |
| `text` | string | Scraped page text, including menus and related-article links (removed in notebook 01) |

The raw article text is not redistributed here: it is large and belongs to the publishers.
To try the pipeline on your own data, save a parquet file with the columns above as
`data/df_news_AI.parquet` and start from notebook 01.

## Files produced by the notebooks

| File | Written by | Read by |
|---|---|---|
| `df_news_clean.parquet` | 01 | 02 |
| `normalized_corpus.pkl` (lemmatized text) | 02 | 02 |
| `coherence_scores.pkl` | 02 | 02 |
| `def_news_AI_clean_topics.parquet` (articles with dominant topic) | 02 | 03, 04 |
| `entity_extraction.pkl` (raw NER output) | 03 | 03 |
| `df_filtered_entities.parquet` (ORG and PRODUCT entities per article) | 03 | 04 |
| `article_sentiment_results_*` (fine-tuned model folders) | 04 | 04 |
| `pred_output_data_set.pkl` (sentiment predictions for the whole corpus) | 04 | 04 |

The slow steps (lemmatization, coherence search, NER inference) run only when their cached file
is missing, so a second run reuses the saved results.

## External data downloaded automatically

* `community-datasets/per_sent` (Hugging Face), used to fine-tune the sentiment model.
* Pretrained models: `all-MiniLM-L6-v2`, `djagatiya/ner-bert-base-cased-ontonotesv5-englishv4`, `distilbert-base-uncased`.
