import torch
from transformers import BertModel, RobertaModel
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder


class TextTransformer(torch.nn.Module):

    def __init__(self, model_name, num_classes, record_embedding: bool = False, no_grad: bool = False, penultimate: bool = False, hidden_size=64, criterion=torch.nn.CrossEntropyLoss()):
        super(TextTransformer, self).__init__()
        model_name = 'bert-base-uncased'
        self.name = model_name
        self.num_classes = num_classes
        # self.train_classes = train_classes
        # self.eval_classes = eval_classes
        # self.dev = dev
        self.criterion = criterion
        self.hidden_size = hidden_size

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        self.penultimate = penultimate


        if 'bert' in model_name:
            self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        else:
            self.bert = RobertaModel.from_pretrained(model_name, return_dict=False)

        # for param in self.bert.parameters():
            # param.requires_grad = False
        
        # self.dropout = torch.nn.Dropout(p=0.3)
        self.dense = torch.nn.Sequential(torch.nn.Linear(self.bert.config.hidden_size, 256), torch.nn.ReLU())
        self.dense2 = torch.nn.Sequential(torch.nn.Linear(256, hidden_size), torch.nn.ReLU())
        self.output = torch.nn.Linear(hidden_size, num_classes)
        # self.output = torch.nn.Linear(self.bert.config.hidden_size, train_classes)

    def params(self):
        for name, param in self.named_params(self):
            yield param
    
    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                # NOTE:
                #print(type(param_t)) #This is Parameter
                #print(type(grad)) # But, this is Tensor!
                tmp = param_t - lr_inner * grad
                #print(tmp)
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    #if first_order:
                    #    grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param): #name = curr_mod_layer
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            # NOTE:
            #setattr(curr_mod, name, param) # Need to convert all the Parameter into Tensor
            curr_mod._parameters[name] = param # Parameter -> Tensor
    ###

    def get_last_layer(self):
        return self.output

    def get_only_bert_output(self, input_ids):
        with torch.no_grad():
            _, bert_output = self.bert(
                input_ids=input_ids,
            )
        return bert_output


    def forward(self, input_ids, bert_output=False):
        with set_grad_enabled(not self.no_grad):
            if not bert_output:
                _, bert_output = self.bert(
                    input_ids=input_ids,
                )
            else:
                bert_output = input_ids
            # output = self.dropout(bert_output)
            output = self.dense(bert_output)
            output_pen = self.dense2(output)
            output = self.embedding_recorder(output_pen)
            output = self.output(output)
            # output = self.output(bert_output)
        if self.penultimate == False:
            return output
        else:
            return output, bert_output