import torch
import torch.nn.functional as F
import copy

from .algorithm import Algorithm
from .modules.utils import put_on_device, empty_context, accuracy

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
        

class Zephyr():

    def __init__(self, device, **kwargs):
        self.device = device
        self.task_counter = 0 
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _deploy(self, messages, test_x, test_y, classes, batched=True):
        predictions = []
        golden = []

        if batched:
            encodeds = []
            longest = -1
            for text, label in zip(test_x, test_y):
                temp_messages = copy.deepcopy(messages)
                temp_messages.append({'role': 'user', 'content': text})
                encoded = self.tokenizer.apply_chat_template(temp_messages, return_tensors="pt", tokenize=True, add_generation_prompt=True)[0]
                longest = max(longest, encoded.shape[0])
                encodeds.append(encoded)
            encoded = None
            for enc in encodeds:
                if enc.shape[0] != longest:
                    enc = torch.cat((torch.IntTensor([self.tokenizer.pad_token_id] * (longest - enc.shape[0])), enc))
                if encoded is None:
                    encoded = enc.reshape((1, -1))
                else:
                    encoded = torch.cat((encoded, enc.reshape((1, -1))))

            generated_ids = self.model.generate(encoded.to(self.device), max_new_tokens=10, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(generated_ids)

            for dec, label in zip(decoded, test_y):
                txt = dec.split('<|assistant|>\n')[-1].strip()

                for idx, cls in enumerate(classes):
                    if cls == label:
                        golden.append(idx)
                        break
                
                found = False
                for idx, cls in enumerate(classes):
                    if cls in txt:
                        predictions.append(idx) 
                        found = True
                        break
                if not found:
                    predictions.append(-1)
        else:
            for text, label in zip(test_x, test_y):
                temp_messages = copy.deepcopy(messages)
                temp_messages.append({'role': 'user', 'content': text})
                encoded = self.tokenizer.apply_chat_template(temp_messages, return_tensors="pt", tokenize=True, add_generation_prompt=True).to(self.device)
                generated_ids = self.model.generate(encoded, max_new_tokens=10, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                decoded = self.tokenizer.batch_decode(generated_ids)[0].split('<|assistant|>\n')[-1]
                
                for idx, cls in enumerate(classes):
                    if cls == label:
                        golden.append(idx)
                        break
                
                found = False
                for idx, cls in enumerate(classes):
                    if cls in decoded:
                        predictions.append(idx) 
                        found = True
                        break
                if not found:
                    predictions.append(-1)
        
        with torch.no_grad():
            acc = accuracy(torch.Tensor(predictions), torch.Tensor(golden))
          
        return acc, predictions, golden
    

    def evaluate(self, classes, task_type, train_x, train_y, test_x, test_y, batched=True, **kwargs):
        options = ""
        for idx, text in enumerate(classes):
            options += f' {idx + 1}) {text}'

        messages = [
            {'role': 'system', 'content': f'Determine {task_type} of the sentence using following options:{options}.'}, 
        ]
        for text, label in zip(train_x, train_y):
            messages.append({'role': 'user', 'content': text})
            messages.append({'role': 'assistant', 'content': label})
        
        acc, predictions, golden = self._deploy(messages, test_x, test_y, classes, batched)

        return acc, predictions, golden