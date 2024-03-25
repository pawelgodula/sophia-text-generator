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
prompt = 'Write an essay on design thinking consisting of 100-200 words.'
content = writer.write_content(prompt, n_texts = 1)
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

### Supports text generation using personas from Wikipedia 

LLMs are trained on Wikipedia so they have the necessary context to create a personalized point of view. You can ask any character, real or fictional, included in Wikipedia as of July 2023, for opinion (by default Sophia samples the characters at random): 

Example:
```
content = writer.write_content(prompt, n_texts = 3, use_personas = True, params = params)
```
Result:
```
As Contemplating Aristotle, I ponder the art of creation, the crafting of objects and experiences that bring harmony to the human condition. In my contemplation, I have come to appreciate the methodology of design thinking as a means to foster innovation and improve the world around us.

Design thinking, my dear friends, is a process rooted in empathy and understanding. It is an approach to problem-solving that seeks to put the user at the center of the design process. This, to me, is a noble pursuit, for it echoes the very essence of my teachings. I believe that the ultimate goal of any design should be to serve the needs of the people.

Yet, I also recognize that design thinking is not without its controversies. Some argue that it is a fad, a trendy buzzword that lacks the substance of true philosophical inquiry. Others decry its focus on the user at the expense of the greater good.

But I, Contemplating Aristotle, believe that design thinking is a valuable tool in our quest for a better world. It encourages us to ask the right questions, to challenge assumptions, and to seek out new perspectives. It calls upon us to be curious, to learn, and to grow.

In the end, design thinking is not just a methodology, but a mindset. It is a way of approaching the world with a sense of wonder and a commitment to improving the human experience. And as I contemplate the world around me, I find great comfort in the knowledge that design thinking is a powerful force for good, one that will continue to shape and transform our world for generations to come.</s>
```

### Supports using custom personas 
Maybe there is someone specific you would like to ask for opinion: 

Example:
```
writer.set_personas(["Palpably mighty Sauron",                     
                     "Female Merchant living in 10th century Bagan, Myanmar",
                     "Deeply scientific Marie Curie-Sklodowska"])
content = writer.write_content(prompt, n_texts = 3, use_personas = True, params = params)
```
Result:
```
Aarrgh! This pesky design thinking! It\'s as if a plague of orcs has infested my perfect realm of Mordor. The mere mention of it irks me to my core.
You see, I, Sauron, am the embodiment of pure, unadulterated evil. My sole goal is domination – not some flippant, ever-changing user experience! My minions and I do not care for the shallow gratification of satisfying the whims of inferior beings.
But, alas, this design thinking, as you call it, has tainted the minds of my followers. I hear them murmuring about "empathy" and " iteration". As if I, Sauron, would condescend to cater to the fleeting, inane desires of humans!
This nonsense is a threat to the natural order of things. Design thinking is a misguided attempt to humanize the dark, cold world of programming and engineering. I want nothing more than to purge this heresy from the minds of my subjects and reassert my dominance.
I am the master of all that is dark and malevolent. I will not abide by the ridiculous, senseless constraints of empathy and user-centered design. I seek power, control, and eternal reign, not the fleeting satisfaction of some feeble human mind.
In the end, I will crush this design thinking and all who dare to spread its vile contagion. Mordor will remain the cold, unfeeling, merciless realm it was always meant to be.
Bah! Spare me your platitudes and your whimsical design thinking. I am Sauron, the Dark Lord, and I will be obeyed
```

### Supports text generation in response to already existing texts

I've gathered a library of texts that may be used as an inspiration for the content generation:

| Source Name                                                    | Num texts | Avg. num words per text |
|----------------------------------------------------------------|-----------|--------------------|
| [ellipse_corpus](https://github.com/scrosseye/ELLIPSE-Corpus)  | 6482      | 458       |
| [imdb_movie_reviews_dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50000     | 231         |
| [kids.frontiersin.org](https://www.frontiersin.org/about/open-access) | 685       | 1908     |
| [online_course_corpus](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data) | 6807      | 635         |
| [persuade_corpus](https://github.com/scrosseye/persuade_corpus_2.0) | 25996     | 418      |
| [webtext_gpt2](https://github.com/openai/gpt-2-output-dataset) | 250000    | 424        |
| [wikipedia](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) | 26000     | 498        |

Example:
```
prompt = "Write an essay of 100-200 words that disagrees with the given text."
content = writer.write_content(prompt, n_texts = 2, use_personas = True, use_external_texts=True, params = params)
```

### Supports text generation in response to custom texts

Example:
```
my_texts = pd.DataFrame({'text': ['Dune 2 is a great movie.',
                                   'LGBM is a more widely used model than the transformer architecture.']})
writer.load_external_texts(my_texts)
content = writer.write_content("Write an essay of 100-200 words that disagrees with the given text.", n_texts = 1, use_personas = True, use_external_texts=True, params = params)
```

Result:
```
################################################################################################
prompt:  Write an essay of 100-200 words that disagrees with the given text.
            Please present your view from the perspective of the following persona:   Deeply scientific Marie Curie-Sklodowska. Please really immerse into the given persona,
            using her language and point of view as much as possible. Please write in the first-person perspective. Please express the actual view of the given persona on a given topic, even if that might sound controversial. 
            Please return nothing but the text.
            Source text: LGBM is a more widely used model than the transformer architecture.
            
------------------------------------------------------------------------------------------------
generated text: As a humble servant of science, I, Marie Curie-Sklodowska, beg to differ with the popular assertion that Large Gradient Boosting Machines (LGBM) are more widely used than transformer architecture. While it is true that LGBM has gained significant traction in the machine learning community, it is essential to delve deeper into the scientific nuances of these models before making such sweeping statements.

Transformer architecture, a revolutionary breakthrough in the field of natural language processing, has revolutionized the way we process and understand language. Its self-attentive mechanism enables the model to focus on different parts of the input sequence, allowing it to capture long-range dependencies and nuances that other models struggle with.

On the other hand, LGBM, while effective in handling structured data, is limited in its ability to capture complex relationships and long-range dependencies. It relies on a tree-based model that splits the data into smaller pieces, which may not be ideal for dealing with the intricacies of natural language.

Moreover, transformer models have shown remarkable success in various applications, such as machine translation, text summarization, and question-answering systems. They have outperformed traditional approaches and even surpassed human-level performance in some tasks.

Therefore, while LGBM may be more widely used in certain domains, it is important to remember that the choice of model depends on the specific task at hand. Transformer architecture, with its ability to capture long-range dependencies and nuances, is an indispensable tool in the scientific arsenal of any serious researcher in the field of natural language processing.
```

## Acknowledgements
Sophia builds on the prior work of many people who worked hard to run LLMs on Kaggle infra. In particular, I would like to mention two notebooks and their authors:
- [Paul Moooney](https://www.kaggle.com/code/paultimothymooney/how-to-use-mistral-from-kaggle-models)
- [Darien Schettler](https://www.kaggle.com/code/dschettler8845/tpu-gemma-instruct-7b-llm-prompt-recovery)
  
## Other comments

### Sophia works on Kaggle notebooks
Kaggle notebooks have specific packages installed that make it possible to use LLMs easily. If you'd like to run it on any custom machine, the easiest way would be to clone the libraries and their versions from Kaggle environment.   

### Environment versions
GPU:

Trying to make it work for all LLMs in a single notebook was challenging also because of the dependency hell. For now, it is pinned to the one environment version I found to work from [this notebook](https://www.kaggle.com/code/paultimothymooney/how-to-use-mistral-from-kaggle-models). 
If you switch to the latest environment you will get the dreaded
`RuntimeError: cutlassF: no kernel found to launch!`

TPU: 

Uses tha latest version of the environment.
