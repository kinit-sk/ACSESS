import torch
from transformers import BertModel, RobertaModel
import torch.nn.functional as F


class TextTransformer(torch.nn.Module):

    def __init__(self, model_name, num_classes, hidden_size=64, criterion=torch.nn.CrossEntropyLoss()):
        super(TextTransformer, self).__init__()
        self.name = model_name
        self.num_classes = num_classes
        # self.train_classes = train_classes
        # self.eval_classes = eval_classes
        # self.dev = dev
        self.criterion = criterion
        self.hidden_size = hidden_size


        if 'bert' in model_name:
            self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        else:
            self.bert = RobertaModel.from_pretrained(model_name, return_dict=False)

        # for param in self.bert.parameters():
            # param.requires_grad = False
        
        self.dropout = torch.nn.Dropout(p=0.3)

        self.dense = torch.nn.Sequential(torch.nn.Linear(self.bert.config.hidden_size, 256), torch.nn.ReLU())
        self.dense2 = torch.nn.Sequential(torch.nn.Linear(256, hidden_size), torch.nn.ReLU())
        self.output = torch.nn.Linear(self.hidden_size, num_classes)
        # self.output = torch.nn.Linear(self.bert.config.hidden_size, train_classes)

    def forward(self, input_ids):
        _, bert_output = self.bert(
          input_ids=input_ids,
          #attention_mask=attention_mask,
          #token_type_ids=token_type_ids
        )
        output = self.dropout(bert_output)

        # output = self.dense(bert_output)
        output = self.dense(output)
        output = self.dense2(output)

        output = self.output(output)
        # output = self.output(bert_output)

        return output