# sophia-text-gen
Generate diverse text datasets with few clicks in Python

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

| Source Name | Num texts | Avg words per text |
|-------------|-----------|--------------------|
| <a href="https://github.com/scrosseye/ELLIPSE-Corpus" target="_blank">ellipse_corpus</a> | 6482 | 458.398488 |
| <a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" target="_blank">imdb_movie_reviews_dataset</a> | 50000 | 231.14594 |
| <a href="https://www.frontiersin.org/about/open-access" target="_blank">kids.frontiersin.org</a> | 685 | 1908.636496 |
| <a href="https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data" target="_blank">online_course_corpus</a> | 6807 | 635.348612 |
| <a href="https://github.com/scrosseye/persuade_corpus_2.0" target="_blank">persuade_corpus</a> | 25996 | 418.129597 |
| <a href="https://github.com/openai/gpt-2-output-dataset" target="_blank">webtext_gpt2</a> | 250000 | 424.414408 |
| <a href="https://www.kaggle.com/datasets/jjinho/wikipedia-20230701" target="_blank">wikipedia</a> | 26000 | 498.335269 |

## Environment
Trying to make it work for all LLMs in a single notebook was challenging also because of the dependency hell. For now it is pinned to the one environment version I found to work from [this notebook](https://www.kaggle.com/code/paultimothymooney/how-to-use-mistral-from-kaggle-models). 

If you switch to the latest environment you will get the dreaded
`RuntimeError: cutlassF: no kernel found to launch!`
