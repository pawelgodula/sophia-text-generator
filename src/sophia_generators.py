import jax
import keras
import keras_nlp
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class sophia_text_generator():
    def __init__(self, model_alias, device = 'gpu', use_quantization = True):
        self.model_alias = model_alias
        self.device = device
        self.use_quanitzation = use_quantization               
        self.default_params = {'max_new_tokens': 64, 'do_sample': True, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}        
        
        self.setup_model(model_alias, device, use_quantization)        
        self.set_personas()
        self.set_moods()
        self.set_mood_levels()
    
    def set_personas(self, custom_personas = None):
        if custom_personas is None:
            self.personas = pd.read_parquet('/kaggle/input/sophia-text-generator/characters_wikipedia.parquet')['title'].values
        else:
            self.personas = custom_personas
    
    def set_moods(self, custom_moods = None):
        if custom_moods is None:
            self.moods = ['Absorbed', 'Aggravated', 'Alert', 'Amused', 'Angry', 'Anxious', 'Apathetic',
                      'Ashamed', 'Awake','Awed', 'Bewildered','Blissful', 'Bored', 'Calm', 'Cautious', 'Cheerful', 'Confident', 'Contemplative', 'Content', 'Contented',
                      'Crushed', 'Curious','Defeated', 'Depressed', 'Desolate', 'Despondent', 'Determined', 'Disappointed', 'Disgusted', 'Disheartened', 'Disillusioned', 'Dismayed',
                      'Drained', 'Drowsy', 'Dynamic', 'Ecstatic', 'Elated', 'Embarrassed', 'Empathetic', 'Energetic', 'Engrossed', 'Enraged', 'Enthusiastic', 'Euphoric',
                      'Excited', 'Exhausted', 'Fascinated', 'Fatigued', 'Fearful', 'Forlorn', 'Frustrated', 'Gloomy', 'Grateful', 'Guilty', 'Happy',
                      'Hopeful', 'Hopeless', 'Horrified', 'Hostile', 'Impressed', 'Indifferent', 'Infuriated',  'Insecure',
                      'Inspired', 'Intrigued', 'Irritated', 'Jealous', 'Joyful', 'Jubilant', 'Lethargic', 'Listless',
                      'Lively', 'Lonely', 'Meditative', 'Melancholic', 'Miserable', 'Nostalgic', 'Offended', 'Optimistic',
                      'Overwhelmed',  'Passive', 'Peppy', 'Perplexed', 'Pessimistic', 'Provoked', 'Puzzled', 'Reflective', 'Refreshed', 'Rejuvenated', 'Resentful',  'Restless',
                      'Sad', 'Sarcastic', 'Satisfied', 'Serene', 'Skeptical', 'Sleepy', 'Spirited', 'Stressed',
                      'Surprised', 'Sympathetic', 'Terrified', 'Thoughtful', 'Tired', 'Vibrant', 'Violated', 'Weary', 'Wonder', 'Zesty']
        else:
            self.moods = custom_moods
            
    def set_mood_levels(self, custom_mood_levels = None):
        if custom_mood_levels is None:
            self.mood_levels = ['Mildly', 'Moderately', 'Noticeably', 'Strongly', 'Intensely', 'Overwhelmingly', 'Extremely']
        else:
            self.mood_levels = custom_mood_levels
            
    def load_external_texts(self, custom_texts = None):
        #custom_texts need to be a pandas df
        if custom_texts is None:
            self.ext_texts = pd.read_parquet('/kaggle/input/sophia-text-generator/sophia_v1.parquet')
        else:
            self.ext_texts = custom_texts            
        print(f'Loaded {self.ext_texts.shape[0]} texts')
    
    def set_model_name(self, model_alias, device): 
        gpu_models = {
            'mistral': "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1",
            'mixtral': '/kaggle/input/mixtral/pytorch/8x7b-instruct-v0.1-hf/1',
            'starling': "berkeley-nest/Starling-LM-7B-alpha",
            'gemma-2b': "/kaggle/input/gemma/transformers/2b/1",
            'gemma-7b': "/kaggle/input/gemma/transformers/7b/1",
            'gemma-2b-it': "/kaggle/input/gemma/transformers/2b-it/1",
            'gemma-7b-it': "/kaggle/input/gemma/transformers/7b-it/1",
        }

        tpu_models = {
            'gemma-2b': "gemma_2b_en",
            'gemma-7b': "gemma_7b_en",
            'gemma-2b-it': "gemma_instruct_2b_en",
            'gemma-7b-it': "gemma_instruct_7b_en",
        }
        
        self.tpu_supported_models = tpu_models.keys()
        
        if device == 'gpu':
            self.model_name = gpu_models.get(model_alias)
            if self.model_name is None:
                supported_models = ", ".join(gpu_models.keys())
                print(f'Unsupported model! Currently we only support the following models on GPU: [{supported_models}]')
        elif device == 'tpu':
            self.model_name = tpu_models.get(model_alias)
            if self.model_name is None:
                supported_models = ", ".join(tpu_models.keys())
                print(f'Unsupported model! Currently we only support the following models on TPU: [{supported_models}]')
        else:
            print(f'Unsupported device! Please choose either GPU or TPU.')    
            
    def setup_model(self, model_alias, device, use_quantization):
        
        self.set_model_name(model_alias, device)
        
        if device == 'gpu':
            if model_alias == 'mixtral':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit= True,
                    bnb_4bit_quant_type= "nf4",
                    bnb_4bit_compute_dtype= torch.bfloat16,
                    bnb_4bit_use_double_quant= True,
                    llm_int8_enable_fp32_cpu_offload= True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True)
            else:
                if use_quantization:
                    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        quantization_config=bnb_config,
                    )
                else:                
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        elif device == 'tpu':
            if model_alias not in self.tpu_supported_models:
                print('Currently we support only the following models on TPU: ', self.tpu_supported_models)
            
            self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.model_name)
            self.tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(self.model_name)

        else: 
            print(f'Unsupported device: {device}')
        print('Model used:', self.model_name)

        
    def build_prompt(self, prompt, use_personas, use_external_texts):
        if use_personas:
            current_persona = np.random.choice(self.personas)
            current_mood = np.random.choice(self.moods)
            current_mood_level = np.random.choice(self.mood_levels) 
            prompt_persona = f"""
            Please present your view from the perspective of the following persona: {current_mood_level} {current_mood} {current_persona}. Please really immerse into the given persona,
            using her languange and point of view as much as possible. Please write in the first person perspective. Please express the actual view of the given persona on a given topic, even if that might sound controversial. 
            Please return nothing but the text."""
            prompt += prompt_persona
        if use_external_texts:
            current_text = np.random.choice(self.ext_texts.text.values)
            prompt_text = f"""
            Source text: {current_text}
            """
            prompt += prompt_text
        
        return prompt
        
    @staticmethod
    def print_content(content):
        for cnt in content:
            print('################################################################################################')
            print('prompt: ', cnt[0])
            print('------------------------------------------------------------------------------------------------')
            print('generated text:', cnt[1])
    
    def strip_prompt(self, text, prompt):
        return text.replace(prompt, '')
    
    def write_with_mistral(self, prompt, params):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            input_ids, 
            **params
                                        )
        response_text = self.tokenizer.batch_decode(output_ids)[0]
        response_text = self.strip_prompt(response_text, prompt)
        try:
            response_text = response_text.split("[/INST] ")[1]
        except:
            pass
        return response_text
    
    def write_with_mixtral(self, prompt, params):
        input_ids = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**input_ids, **params)
        response_text = self.tokenizer.batch_decode(generated_ids)[0]
        response_text = self.strip_prompt(response_text, prompt)
        return response_text
    
    def write_with_starling(self, prompt, params):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **params,
                                            )[0]
        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        response_text = self.strip_prompt(response_text, prompt)
        try:
            response_text = response_text.split('\n\n---\n\n')[1]
        except:
            pass

        return response_text
    
    def write_with_gemma(self, prompt, params):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if 'gemma-7b' in self.model_alias:
            params['do_sample'] = False
        outputs = self.model.generate(**input_ids, **params)
        response_text = self.tokenizer.decode(outputs[0])
        response_text = self.strip_prompt(response_text, prompt)
        return response_text          
    
    def write_with_gemma_tpu(self, prompt, params):
        sampler_kwargs = {'p': params['top_p'], 'k': params['top_k'], 'temperature': params['temperature']}
        sampler = keras_nlp.samplers.TopPSampler(**sampler_kwargs)
        input_text_length = self.tokenizer.tokenize(prompt).shape[0]
        self.model.compile(sampler = sampler)
        response_text = self.model.generate(
                inputs=prompt,         
                max_length = input_text_length + params['max_new_tokens']
            )
        response_text = self.strip_prompt(response_text, prompt)
        return response_text
    
    def write_content(self, prompt, n_texts = 1, params = None, use_personas = False, use_external_texts = False):
        if self.model_alias == 'mistral':
            writing_function = self.write_with_mistral
        elif self.model_alias == 'mixtral':
            writing_function = self.write_with_mixtral
        elif self.model_alias == 'starling':
            writing_function = self.write_with_starling
        elif 'gemma' in self.model_alias:
            if self.device == 'gpu':
                writing_function = self.write_with_gemma
            elif self.device == 'tpu':
                writing_function = self.write_with_gemma_tpu
        if params is None:  
            params = self.default_params 
        
        outputs = []            
        for _ in range(n_texts):
            final_prompt = self.build_prompt(prompt, use_personas, use_external_texts)
            outputs.append((final_prompt, writing_function(final_prompt, params)))
        return outputs
