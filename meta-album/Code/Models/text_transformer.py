import torch
from transformers import BertModel, RobertaModel
import torch.nn.functional as F


class TextTransformer(torch.nn.Module):

    def __init__(self, model_name, dev, train_classes, eval_classes=None , trainable=False, hidden_size=64, criterion=torch.nn.CrossEntropyLoss()):
        super(TextTransformer, self).__init__()
        self.name = model_name
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.dev = dev
        self.criterion = criterion
        self.hidden_size = hidden_size


        if 'bert' in model_name:
            self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        else:
            self.bert = RobertaModel.from_pretrained(model_name, return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dense = torch.nn.Sequential(torch.nn.Linear(self.bert.config.hidden_size, 256), torch.nn.ReLU())
        self.dense2 = torch.nn.Sequential(torch.nn.Linear(256, hidden_size), torch.nn.ReLU())
        self.output = torch.nn.Linear(hidden_size, train_classes)

    def compute_in_features(self, input_ids):
        _, bert_output = self.bert(
          input_ids=input_ids,
        )
        output = self.dense(bert_output)
        output = self.dense2(output)

        return output

    def forward(self, input_ids):
        _, bert_output = self.bert(
          input_ids=input_ids,
        )
        output = self.dense(bert_output)
        output = self.dense2(output)
        output = self.output(output)

        return output

    def forward_weights(self, input_ids, weights, embedding=False):
        with torch.no_grad():
            _, bert_output = self.bert(
            input_ids=input_ids,
            )

        out = F.relu(F.linear(bert_output, weight=weights[-6], bias=weights[-5]))
        out = F.relu(F.linear(out, weight=weights[-4], bias=weights[-3]))
        if embedding:
            return out
        out = F.linear(out, weight=weights[-2], bias=weights[-1])
        return out

    def modify_out_layer(self, num_classes):
        if num_classes is None:
            num_classes = self.eval_classes
        self.output = torch.nn.Linear(in_features=self.hidden_size, out_features=num_classes).to(self.dev)
        self.output.bias = torch.nn.Parameter(torch.zeros(self.output.bias.size(), device=self.dev))

    def freeze_layers(self, freeze, num_classes):
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.modify_out_layer(num_classes)

    def load_params(self, state_dict):
        del state_dict['output.weight']
        del state_dict['output.bias']
        self.load_state_dict(state_dict, strict=False)