# sophia-text-generator
Easily generate diverse text datasets on Kaggle notebooks with the latest & greatest open-source LLMs

## Idea
Recently all major Kaggle LLM/NLP competitions rely on participants to create a training dataset. The goal of Sophia is to enable everyone to build their own training dataset on Kaggle infra. In various competitions, it was shown that datasets of 2k-3k samples make a meaningful impact on both CV and LB scores.
The tutorial notebooks are here:
- [GPU Sophia tutorial](https://www.kaggle.com/code/narsil/sophia-text-generator-tutorial-gpu)
- [TPU Sophia tutorial](https://www.kaggle.com/code/narsil/sophia-text-generator-tutorial-tpu)

## Key features: 

### 0. It is simple

```
from sophia_generators import sophia_text_generator
writer = sophia_text_generator("gemma-7b-it", device = 'tpu')
texts = writer.write_content('Write an essay on design thinking consisting of 100-200 words.', n_texts = 1)
```

### 1. Supports GPU and TPU
To maximize quota utilization

### 2. Supports the best open source models (according to [LLM arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)) that fit into the Kaggle infra:

GPU:
- Mixtral 8 X 7B
- Mistral 7B v0.1
- Mistral 7B v0.2
- Starling 7B
- Gemma 2B

TPU:
- Gemma 2B & 7B

### 3. Supports text generation using personas from Wikipedia (all characters, real or fictional, described in Wikipedia)
How would you 

### 4. Supports text generation in response to already existing texts

#### Datasets included 

| Source Name                                                    | Num texts | Avg. num words per text |
|----------------------------------------------------------------|-----------|--------------------|
| [ellipse_corpus](https://github.com/scrosseye/ELLIPSE-Corpus)  | 6482      | 458       |
| [imdb_movie_reviews_dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50000     | 231         |
| [kids.frontiersin.org](https://www.frontiersin.org/about/open-access) | 685       | 1908     |
| [online_course_corpus](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data) | 6807      | 635         |
| [persuade_corpus](https://github.com/scrosseye/persuade_corpus_2.0) | 25996     | 418      |
| [webtext_gpt2](https://github.com/openai/gpt-2-output-dataset) | 250000    | 424        |
| [wikipedia](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) | 26000     | 498        |

## Acknowledgements
Sophia builds on the prior work of many people who worked hard to run LLMs on Kaggle infra. In particular, I would like to mention two notebooks and their authors:

## Other comments

### Sophia works on Kaggle notebooks
Kaggle notebooks have specific packages installed that make it possible to use LLMs easily. If you'd like to run it on any custom machine, the easiest way would be to clone the libraries and their versions from Kaggle environment.   

## Environment
Trying to make it work for all LLMs in a single notebook was challenging also because of the dependency hell. For now, it is pinned to the one environment version I found to work from [this notebook](https://www.kaggle.com/code/paultimothymooney/how-to-use-mistral-from-kaggle-models). 

If you switch to the latest environment you will get the dreaded
`RuntimeError: cutlassF: no kernel found to launch!`
