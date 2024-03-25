# sophia-text-generator
Easily generate diverse text datasets on Kaggle notebooks with the latest & greatest open-source LLMs

## Idea
Recently all major Kaggle LLM/NLP competitions rely on participants to create a training dataset. The goal of Sophia is to enable everyone to build their own training dataset on Kaggle infra. In various competitions, it was shown that datasets of 2k-3k samples make a meaningful impact on both CV and LB scores.
The tutorial notebooks are here:
- [GPU Sophia tutorial](https://www.kaggle.com/code/narsil/sophia-text-generator-tutorial-gpu)
- [TPU Sophia tutorial](https://www.kaggle.com/code/narsil/sophia-text-generator-tutorial-tpu)

## Key features: 

### It is simple

```
from sophia_generators import sophia_text_generator
writer = sophia_text_generator("gemma-7b-it", device = 'tpu')
texts = writer.write_content('Write an essay on design thinking consisting of 100-200 words.', n_texts = 1)
```

### Supports GPU and TPU
To maximize quota utilization

### Supports the best open source models (according to [LLM arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)) that fit into the Kaggle infra:

GPU:
- Mixtral 8 X 7B
- Mistral 7B it v0.1
- Mistral 7B it v0.2
- Starling 7B
- Gemma 2B
- Gemma 2B it

TPU:
- Gemma 2B
- Gemma 2B it
- Gemma 7B
- Gemma 7B it

### Supports text generation using personas from Wikipedia (all characters, real or fictional, described in Wikipedia)
LLMs are trained on Wikipedia so they have the necessary context to create personalized point of view.
```
content = writer.write_content(prompt, n_texts = 3, use_personas = True, params = params)
```
Result:
```
As Guy Degrenne, the Overwhelmingly Disheartened, I've spent years observing the design thinking community with a heavy heart. I've watched as they've embraced empathy, ideation, and prototyping with open arms. I've listened to their passionate speeches about creating user-centered solutions to real-world problems. But as I look around, I can't help but feel that design thinking, with all its promise, has failed us.

Empathy, they say, is the foundation of design thinking. But how can we truly understand the complexities of human emotions and experiences through a few interviews and observations? How can we reduce the intricacies of a problem to a neat, tidy design solution?

Ideation, they tell us, is about generating a multitude of ideas. But isn't that just a fancy way of saying brainstorming? And how many brilliant ideas have been lost in the pursuit of the most feasible or profitable solution?

Prototyping, they claim, is about testing and refining our ideas. But what happens when our prototypes don't solve the root cause of the problem? What happens when they merely scratch the surface, providing a temporary band-aid to a deep-rooted issue?

Design thinking, I believe, is a futile attempt to solve real-world problems. It's a reductionist approach that oversimplifies the complexities of human experience. It's a process that prioritizes convenience and profitability over genuine impact.
```
### Supports text generation in response to already existing texts

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
