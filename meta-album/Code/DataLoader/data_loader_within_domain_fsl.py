# code inspired from https://github.com/yinboc/prototypical-network-pytorch
import os.path as osp
import pandas as pd
import numpy as np
import torch
import os
import math

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FewshotDataset(Dataset):

    def __init__(self, img_path, md_path, label_col, file_col, allowed_labels, 
                 img_size=128):
        # path to image folder
        self.img_path = img_path
        # path to meta-data file
        self.md_path = md_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.file_col = file_col
        # labels to use  
        self.allowed_labels = allowed_labels

        self.md = pd.read_csv(self.md_path)
        # select only data with permissible labels
        self.md = self.md[self.md[label_col].isin(self.allowed_labels)]

        label_to_id = dict(); id = 0
        self.img_paths = np.array([osp.join(self.img_path, x) for x in 
            self.md[file_col]])
        # transform string labels into non-negative integer IDs
        self.labels = []
        for label in self.md[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)
        self.transform = transforms.Compose([transforms.Resize((img_size,
            img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()

class FewshotTextDataset(Dataset):

    def __init__(self, md_path, label_col, text_col, allowed_labels, max_len=50, tokenizer=None):
        # path to meta-data file
        self.md_path = md_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.text_col = text_col
        # labels to use  
        self.allowed_labels = allowed_labels

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.md = pd.read_csv(self.md_path)
        # select only data with permissible labels
        self.md = self.md[self.md[label_col].isin(self.allowed_labels)]

        label_to_id = dict(); id = 0
        self.texts = [text for text in self.md[text_col]]
        # transform string labels into non-negative integer IDs
        self.labels = []
        for label in self.md[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text, label = self.texts[i], self.labels[i]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_tensors='pt')
        ids = encoded['input_ids'][0]
        return ids, torch.LongTensor([label]).squeeze()


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            if ind.shape[0] < n_per:
                ind = np.repeat(ind, math.floor(n_per / ind.shape[0]) + 1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])

            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class TestFewshotDataset(Dataset):

    def __init__(self, img_path, support_path, query_path, label_col, file_col, img_size=128):
        # path to image folder
        self.img_path = img_path
        # path to meta-data file
        self.support_path = support_path
        self.query_path = query_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.file_col = file_col

        self.support = pd.read_csv(self.support_path)
        self.query = pd.read_csv(self.query_path)
        # select only data with permissible labels

        label_to_id = dict(); id = 0
        self.img_paths = np.array([osp.join(self.img_path, x) for x in self.query[file_col]])
        self.support_img_paths = np.array([osp.join(self.img_path, x) for x in self.support[file_col]])
        # transform string labels into non-negative integer IDs
        self.support_labels = []
        self.query_labels = []
        for label in self.query[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.query_labels.append(label_to_id[label])
        self.query_labels = np.array(self.query_labels)

        for label in self.support[label_col]:
            self.support_labels.append(label_to_id[label])
        self.support_labels = np.array(self.support_labels)

        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        if i < 0:
            path, label = self.support_img_paths[i * -1], self.support_labels[i * -1]
        else:
            path, label = self.img_paths[i], self.query_labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()


class TestFewshotTextDataset(Dataset):

    def __init__(self, support_path, query_path, label_col, text_col, max_len=50, tokenizer=None):
        # path to meta-data file
        self.support_path = support_path
        self.query_path = query_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.text_col = text_col

        self.support = pd.read_csv(self.support_path)
        self.query = pd.read_csv(self.query_path)
        # select only data with permissible labels
        self.max_len = max_len
        self.tokenizer = tokenizer

        label_to_id = dict(); id = 0
        self.texts = [text for text in self.query[text_col]]
        self.support_texts = [text for text in self.support[text_col]]
        # transform string labels into non-negative integer IDs
        self.support_labels = []
        self.query_labels = []
        for label in self.query[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.query_labels.append(label_to_id[label])
        self.query_labels = np.array(self.query_labels)

        for label in self.support[label_col]:
            self.support_labels.append(label_to_id[label])
        self.support_labels = np.array(self.support_labels)


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        if i < 0:
            text, label = self.support_texts[i * -1], self.support_labels[i * -1]
        else:
            text, label = self.texts[i], self.query_labels[i]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_tensors='pt')
        ids = encoded['input_ids'][0]
        return ids, torch.LongTensor([label]).squeeze()


class TestFewshotDatasetSetSupport(Dataset):

    def __init__(self, img_path, support_set, query_path, label_col, file_col, img_size=128):
        # path to image folder
        self.img_path = img_path
        # path to meta-data file
        self.query_path = query_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.file_col = file_col

        self.support = support_set
        self.query = pd.read_csv(self.query_path)
        # select only data with permissible labels

        label_to_id = dict(); id = 0
        self.img_paths = np.array([osp.join(self.img_path, x) for x in self.query[file_col]])
        self.support_img_paths = np.array([osp.join(self.img_path, x) for x in self.support[file_col]])
        # transform string labels into non-negative integer IDs
        self.support_labels = []
        self.query_labels = []
        for label in self.query[label_col]:
            if not label in label_to_id:
                label_to_id[label] = id
                id += 1
            self.query_labels.append(label_to_id[label])
        self.query_labels = np.array(self.query_labels)

        for label in self.support[label_col]:
            self.support_labels.append(label_to_id[label])
        self.support_labels = np.array(self.support_labels)

        self.transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        if i < 0:
            path, label = self.support_img_paths[i * -1], self.support_labels[i * -1]
        else:
            path, label = self.img_paths[i], self.query_labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()

class TestFewshotICLDataset(Dataset):

    def __init__(self, support_path, query_path, label_col, text_col, dataset):
        # path to meta-data file
        self.support_path = support_path
        self.query_path = query_path
        # column of meta-data file that contains label information
        self.label_col = label_col
        self.text_col = text_col

        self.support = pd.read_csv(self.support_path)
        self.query = pd.read_csv(self.query_path)

        self.dataset = dataset

        self.mapper = {
            '20News': {
                'comp.sys.ibm.pc.hardware': 'IBM', 
                'talk.politics.mideast': 'Middle East Politics',
                'comp.windows.x': 'Windows XP', 
                'rec.motorcycles': 'Motorcycles', 
                'sci.med': 'Medicine', 
                'misc.forsale': 'For Sale',
                'talk.religion.misc': 'Religion', 
                'comp.os.ms-windows.misc': 'MS Windows',
                'rec.sport.baseball': 'Baseball', 
                'rec.autos': 'Auto', 
                'rec.sport.hockey': 'Hockey',
                'comp.sys.mac.hardware': 'Mac', 
                'comp.graphics': 'Graphics', 
                'soc.religion.christian': 'Christianity',
                'talk.politics.guns': 'Guns', 
                'sci.electronics': 'Electronics', 
                'sci.space': 'Space', 
                'sci.crypt': 'Crypto',
                'alt.atheism': 'Atheism', 
                'talk.politics.misc': 'Politics', 
                'None': 'None'
            },
            'NewsCategory': {
                'POLITICS': 'Politics', 
                'WORLD NEWS': 'World News', 
                'PARENTING': 'Parenting', 
                'MONEY': 'Money', 
                'WELLNESS': 'Wellness',
                'BUSINESS': 'Business', 
                'WEDDINGS': 'Weddings', 
                'ENTERTAINMENT': 'Entertainment', 
                'IMPACT': 'Impact', 
                'BLACK VOICES': 'Black Voices',
                'QUEER VOICES': 'Queer Voices', 
                'CRIME': 'Crime', 
                'DIVORCE': 'Divorce', 
                'FOOD & DRINK': 'Food and Drink', 
                'WORLDPOST': 'Worldpost',
                'PARENTS': 'Parents', 
                'TRAVEL': 'Travel', 
                'THE WORLDPOST': 'The Worldpost', 
                'HEALTHY LIVING': 'Healthy Living', 
                'TASTE': 'Taste',
                'MEDIA': 'Media', 
                'CULTURE & ARTS': 'Culture and Arts', 
                'STYLE': 'Style', 
                'WEIRD NEWS': 'Weird News', 
                'STYLE & BEAUTY': 'Style and Beauty',
                'COMEDY': 'Comedy', 
                'HOME & LIVING': 'Home and Living', 
                'SPORTS': 'Sports', 
                'ENVIRONMENT': 'Environment', 
                'EDUCATION': 'Education',
                'GOOD NEWS': 'Good News', 
                'FIFTY': 'Fifty', 
                'WOMEN': 'Women', 
                'SCIENCE': 'Science', 
                'ARTS & CULTURE': 'Arts and Culture',
                'U.S. NEWS': 'US News', 
                'GREEN': 'Green', 
                'TECH': 'Tech', 
                'RELIGION': 'Religion', 
                'LATINO VOICES': 'Latino Voices',
                'COLLEGE': 'College', 
                'ARTS': 'Arts'
            },
            'atis': {
                'atis_flight':          'Flight',
                'atis_abbreviation':    'Abbreviation',
                'atis_city':            'City',
                'atis_airfare':         'Airfare',
                'atis_ground_service':  'Ground Service',
                'atis_ground_fare':     'Ground Fare',
                'atis_airline':         'Airline',
                'atis_flight_no':       'Flight Number',
                'atis_aircraft':        'Aircraft',
                'atis_distance':        'Distance',
                'atis_capacity':        'Capacity',
                'atis_flight_time':     'Flight Time',
                'atis_quantity':        'Quantity',
                'atis_airport':         'Airport',
            },
            'facebook': {
                'get_info_road_condition':  'Road Condition',
                'get_estimated_departure':  'Departure',
                'get_estimated_duration':   'Duration',
                'get_event':                'Event',
                'get_estimated_arrival':    'Arrival',
                'get_directions':           'Directions',
                'get_distance':             'Distance',
                'get_info_traffic':         'Traffic',
                'update_directions':        'Update',
                'combine':                  'Combine',
                'get_info_route':           'Route',
                'get_location':             'Location',
            },
            'liu': {
                'alarm_set': 'Set Alarm',
                'qa_definition': 'Definition',
                'play_music': 'Play Music',
                'calendar_set': 'Set Calendar',
                'play_radio': 'Play Radio',
                'general_confirm': 'Confirm',
                'general_quirky': 'Quirky',
                'email_sendemail': 'Send Email',
                'qa_currency': 'Currency',
                'music_likeness': 'Like',
                'social_query': 'Social Query',
                'news_query': 'News',
                'general_dontcare': 'Neutral',
                'general_commandstop': 'Stop',
                'calendar_query': 'Calendar Query',
                'alarm_query': 'Alarm Query',
                'recommendation_movies': 'Recommend Movie',
                'play_audiobook': 'Play Audiobook',
                'weather_query': 'Weather Query',
                'general_praise': 'Praise',
                'social_post': 'Social Post',
                'play_podcasts': 'Play Podcast',
                'general_negate': 'Negate',
                'cooking_recipe': 'Recipe',
                'general_affirm': 'Affirm',
                'calendar_remove': 'Remove Calendar',
                'recommendation_locations': 'Recommend Location',
                'lists_remove': 'Remove List',
                'general_explain': 'Explain',
                'email_query': 'Email Query',
                'qa_maths': 'Maths',
                'qa_stock': 'Stock',
                'transport_query': 'Transport Query',
                'takeaway_order': 'Order Takeaway',
                'datetime_query': 'Datetime Query',
                'general_repeat': 'Repeat',
                'qa_factoid': 'Factoid',
                'transport_taxi': 'Taxi',
                'lists_createoradd': 'Add List',
                'email_addcontact': 'Add Contact',
                'recommendation_events': 'Recommend Event',
                'audio_volume_mute': 'Mute Volume',
                'lists_query': 'List Query',
                'transport_ticket':' Ticket',
                'datetime_convert': 'Convert Datetime',
                'general_joke': 'Joke',
                'alarm_remove': 'Remove Alarm',
                'transport_traffic': 'Traffic',
                'audio_volume_up': 'Volume Up',
                'takeaway_query': 'Takeaway Query',
                'play_game': 'Play Game',
                'email_querycontact': 'Contact Query',
                'music_query': 'Music Query',
                'audio_volume_other': 'Volume Other',
                'audio_volume_down': 'Volume Down',
                'music_settings': 'Music Setting',
                'general_greet': 'Greet',
                'music_dislikeness': 'Dislike',
            },
            'snips': {
                'AddToPlaylist':            'Playlist',
                'GetWeather':               'Weather',
                'SearchScreeningEvent':     'Event',
                'PlayMusic':                'Musing',
                'SearchCreativeWork':       'Creative Work',
                'RateBook':                 'Rate Book',
                'BookRestaurant':           'Book Restaurant',
            },
        }[dataset]

        if dataset in ['20News', 'NewsCategory']:
            self.task_type = 'category'
        else:
            self.task_type = 'intent'

        

        label_to_id = dict()
        id = 0

        self.texts = [text for text in self.query[text_col]]
        self.support_texts = [text for text in self.support[text_col]]
        # transform string labels into non-negative integer IDs
        self.support_labels = []
        self.query_labels = []
        for label in self.query[label_col]:
            if label not in label_to_id:
                label_to_id[label] = id
                id += 1
            self.query_labels.append(label_to_id[label])
        self.query_labels = np.array(self.query_labels)

        for label in self.support[label_col]:
            self.support_labels.append(label_to_id[label])
        self.support_labels = np.array(self.support_labels)

        self.label_to_id = label_to_id
        self.id_to_label = {v:self.mapper[k] for k, v in label_to_id.items()}


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        if i < 0:
            text, label = self.support_texts[i * -1], self.support_labels[i * -1]
        else:
            text, label = self.texts[i], self.query_labels[i]
        return text, self.id_to_label[label]

    def get_class_names_and_task(self):
        return self.task_type


class TestCategoriesICLSampler():

    def __init__(self, support_label, query_label, n_batch, n_cls, support_size, query_size):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.support_size = support_size
        self.query_size = query_size

        query_label = np.array(query_label)
        support_label = np.array(support_label)
        self.support_inds = []
        self.query_inds = []
        for i in range(max(query_label) + 1):
            support_ind = np.argwhere(support_label == i).reshape(-1)
            query_ind = np.argwhere(query_label == i).reshape(-1)
            if support_ind.shape[0] < support_size:
                print('Repeating support')
                support_ind = np.repeat(support_ind, math.floor(support_size / support_ind.shape[0]) + 1)
            if query_ind.shape[0] < query_size:
                print('Repeating query')
                query_ind = np.repeat(query_ind, math.floor(query_size / query_ind.shape[0]) + 1)
            support_ind = torch.from_numpy(support_ind)
            query_ind = torch.from_numpy(query_ind)
            self.support_inds.append(support_ind)
            self.query_inds.append(query_ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.query_inds))[:self.n_cls]

            for c in classes:
                supp = self.support_inds[c]
                quer = self.query_inds[c]
                supp_pos = torch.randperm(len(supp))[:self.support_size]
                quer_pos = torch.randperm(len(quer))[:self.query_size]
                supp_dat = supp[supp_pos]
                quer_dat = quer[quer_pos]
                if self.support_size != 0:
                    dat = torch.cat((supp_dat * -1, quer_dat)).reshape(-1)
                else:
                    dat = quer_dat.reshape(-1)
                batch.append(dat)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class TestCategoriesSampler():

    def __init__(self, support_label, query_label, n_batch, n_cls, support_size, query_size):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.support_size = support_size
        self.query_size = query_size

        query_label = np.array(query_label)
        support_label = np.array(support_label)
        self.support_inds = []
        self.query_inds = []
        for i in range(max(query_label) + 1):
            support_ind = np.argwhere(support_label == i).reshape(-1)
            query_ind = np.argwhere(query_label == i).reshape(-1)
            if support_ind.shape[0] < support_size:
                print('Repeating support')
                support_ind = np.repeat(support_ind, math.floor(support_size / support_ind.shape[0]) + 1)
            if query_ind.shape[0] < query_size:
                print('Repeating query')
                query_ind = np.repeat(query_ind, math.floor(query_size / query_ind.shape[0]) + 1)
            support_ind = torch.from_numpy(support_ind)
            query_ind = torch.from_numpy(query_ind)
            self.support_inds.append(support_ind)
            self.query_inds.append(query_ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.query_inds))[:self.n_cls]
            for c in classes:
                supp = self.support_inds[c]
                quer = self.query_inds[c]
                supp_pos = torch.randperm(len(supp))[:self.support_size]
                quer_pos = torch.randperm(len(quer))[:self.query_size]
                supp_dat = supp[supp_pos]
                quer_dat = quer[quer_pos]
                dat = torch.cat((supp_dat * -1, quer_dat)).reshape(-1)
                batch.append(dat)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch



def process_labels(batch_size, num_classes):
    return torch.arange(num_classes).repeat(batch_size//num_classes).long()


def extract_info(data_dir, dataset):
    info = {"img_path": f"{data_dir}/{dataset}/images/",
            "metadata_path": f"{data_dir}/{dataset}/labels.csv",
            "label_col": "CATEGORY",
            "file_col": "FILE_NAME"}
    return info["label_col"], info["file_col"],\
           info["img_path"], info["metadata_path"]

def extract_text_info(data_dir, dataset):
    return 'text', 'label', f'{data_dir}/{dataset}/train.csv'

def extract_train_valid_test_info(data_dir, dataset):
    path = os.path.join(data_dir, dataset)
    return os.path.join(path, 'train.csv'), os.path.join(path, 'valid.csv'), os.path.join(path, 'test.csv'),

def extract_train_valid_test_info_any_shot(data_dir, dataset, strategy, k):
    path = os.path.join(data_dir, dataset)
    return os.path.join(path, strategy, f'{k if k > 0 else 1}.csv'), os.path.join(path, 'valid.csv'), os.path.join(path, 'test.csv'),


def extract_strategy_samples_info(data_dir, dataset, strategy, run, sub_strategy='basic'):
    if strategy == 'basic':
        if sub_strategy == 'basic':
            path = os.path.join(data_dir, dataset, 'train.csv')
        else:
            path = os.path.join(data_dir, dataset, sub_strategy, 'train.csv')
    else:
        if sub_strategy == 'basic':
            if run != -1:
                path = os.path.join(data_dir, dataset, strategy, f'{strategy}_{run}.csv')
            else:
                path = os.path.join(data_dir, dataset, strategy, f'{strategy}.csv')
        else:
            if run != -1:
                path = os.path.join(data_dir, dataset, sub_strategy, strategy, f'{strategy}_{run}.csv')
            else:
                path = os.path.join(data_dir, dataset, sub_strategy, strategy, f'{strategy}.csv')
    return path

def extract_cartography_change_samples_info(data_dir, dataset, strategy, run, sub_strategy='basic'):
    path = os.path.join(data_dir, dataset, sub_strategy, f'{strategy}.csv')
    return path



def train_test_split_classes(classes, test_split=0.3):
    classes = np.unique(np.array(classes))
    np.random.shuffle(classes)
    cut_off = int((1-test_split)*len(classes))
    return classes[:cut_off], classes[cut_off:]
