# sophia-text-gen
Generate diverse text datasets with few clicks in Python on Kaggle notebooks

## Idea
Simple generation of diverse texts on Kaggle notebooks with the latest & greatest open-source LLMs.

### Models
I integrated some of the best open source models from [LLM arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) that fit into Kaggle infra.
- Mixtral 8 X 7B
- Starling 7B
- Mistral 7B
- Gemma 2B & 7B

Recently all major LLM Kaggle competitions rely on participants to create a training dataset. I find more & more Kagglers are sharing great datasets - so I thought I would build something to enable everyone to build their training dataset on Kaggle infra. Let's unleash everyone's prompt creativity!

- Simplicity & Performance: Sophia is a wrapper for text generation with the top Open Source LLMs from [LLM arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) that fit into Kaggle infra and exist on HF. For now it supports the following models:
-
- It builds on a great work of many Kagglers who shown how to integrate particular models:
- Create diversity in train datasets for Kaggle competitions by:
- Pre-existing library of texts (see below in Datasets included)
- Providing 

## Datasets included 

| Source Name                                                    | Num texts | Avg. num words per text |
|----------------------------------------------------------------|-----------|--------------------|
| [ellipse_corpus](https://github.com/scrosseye/ELLIPSE-Corpus)  | 6482      | 458       |
| [imdb_movie_reviews_dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50000     | 231         |
| [kids.frontiersin.org](https://www.frontiersin.org/about/open-access) | 685       | 1908     |
| [online_course_corpus](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data) | 6807      | 635         |
| [persuade_corpus](https://github.com/scrosseye/persuade_corpus_2.0) | 25996     | 418      |
| [webtext_gpt2](https://github.com/openai/gpt-2-output-dataset) | 250000    | 424        |
| [wikipedia](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) | 26000     | 498        |

## Environment
Trying to make it work for all LLMs in a single notebook was challenging also because of the dependency hell. For now it is pinned to the one environment version I found to work from [this notebook](https://www.kaggle.com/code/paultimothymooney/how-to-use-mistral-from-kaggle-models). 

If you switch to the latest environment you will get the dreaded
`RuntimeError: cutlassF: no kernel found to launch!`
