import torch
import torch.nn.functional as F

from .algorithm import Algorithm
from .modules.utils import put_on_device, empty_context, accuracy
                           
                           
class MatchingNetwork(Algorithm):
    
    def __init__(self, meta_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.meta_batch_size = meta_batch_size

        self.task_counter = 0 
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in 
            self.baselearner.parameters()]

        for p in self.initialization:
            p.requires_grad = True
                
        self.optimizer = self.opt_fn(self.initialization, lr=self.lr)
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, 
                num_classes=None):
        if train_mode:
            contxt = empty_context
            num_classes = self.baselearner.train_classes
        else:
            contxt = torch.no_grad
            if num_classes is None:
                num_classes = self.baselearner.eval_classes

        with contxt():
            support_embeddings = self.baselearner.forward_weights(train_x, 
                self.initialization, embedding=True)
            query_embeddings = self.baselearner.forward_weights(test_x, 
                self.initialization, embedding=True)
            
            s_norm = support_embeddings / support_embeddings.norm(dim=1
                ).unsqueeze(1)
            q_norm = query_embeddings / query_embeddings.norm(dim=1
                ).unsqueeze(1)
            cosine_similarities = torch.mm(s_norm, q_norm.transpose(0,1)).t()

            y = torch.zeros((len(train_x), num_classes), 
                device=self.initialization[0].device)
            y[torch.arange(len(train_x)), train_y] = 1

            out = torch.mm(cosine_similarities, y)
            loss = self.baselearner.criterion(out, test_y) 
            
        with torch.no_grad():
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = accuracy(preds, test_y)
            
        return acc, loss, probs.cpu().numpy(), preds.cpu().numpy()
    
    def train(self, train_x, train_y, test_x, test_y):
        self.baselearner.train()
        self.task_counter += 1
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
            [train_x, train_y, test_x, test_y])
        
        acc, loss, probs, preds = self._deploy(train_x, train_y, test_x, 
            test_y, True)
        loss.backward()
        if self.task_counter % self.meta_batch_size == 0: 
            self.optimizer.step()  
            self.optimizer.zero_grad()
            
        return acc, loss.item(), probs, preds

    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 **kwargs):
        self.baselearner.eval()
        train_x, train_y, test_x, test_y = put_on_device(self.dev,
            [train_x, train_y, test_x, test_y])
            
        acc, loss, probs, preds = self._deploy(train_x, train_y, test_x, 
            test_y, False, num_classes)

        return acc, [loss.item()], probs, preds
    
    def dump_state(self):
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
