import os
import json
import torch

def prepro_sentence(test_inputs, max_length, bos_token_id, eos_token_id):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        ids = [bos_token_id] + test_input + [eos_token_id]
        n_mask = max_length-len(ids)
        input_ids.append(ids+[0 for _ in range(n_mask)])
        attention_mask.append([1 for _ in ids] +
                              [0 for _ in range(n_mask)])
        token_type_ids.append([1 for _ in ids] +
                              [0 for _ in range(n_mask)])

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}


def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id, negate=False,
                                allow_truncation=False):

    assert not negate

    if bos_token_id is not None:
        ids1 = [bos_token_id] + ids1
    if eos_token_id is not None:
        ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    if negate:
        token_type_ids = [0 for _ in ids1] + [-1 for _ in ids2] + [0 for _ in range(n_mask)]
    else:
        token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    input_length = len(ids1)
    output_length = len(ids2)
    return input_ids, attention_mask, token_type_ids, input_length, output_length


def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    input_lengths = []
    output_lengths = []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids, _input_length, _output_length = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)
            input_lengths.append(_input_length)
            output_lengths.append(_output_length)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "input_length":torch.LongTensor(input_lengths),
            "output_length":torch.LongTensor(output_lengths)
            }

def flatten_label_losses(label_losses, dev_data):
    for label in range(len(label_losses)):
        k = int(len(label_losses[label]) / len(dev_data))
        label_losses[label] = [
            label_losses[label][k*i:k*(i+1)]
            for i in range(len(dev_data))]
    return label_losses

# get templates + verbalizers
def get_prompts(task, idx):

    if task in ["SST-2", "sst-5", "mr", "cr"]:
        templates = ["A %s one . ", "It was %s . ",
                     "All in all %s . ", "A %s piece . "]
    elif task in ["yelp_full", "yelp_binary", "amazon", 'amazon_b']:
        templates = ["A %s one. ", "It was %s. ",
                     "All in all %s. ", "A %s piece. "]
    elif task=="trec":
        templates = ["%s : ", "Q: %s : ", "Why %s ? ", "Answer: %s . "]
    elif task in ["agnews", "sogou", "dbpedia", "yahoo"]:
        templates = ["Topic: %s. ", "Subject: %s. ",
                     "This is about %s. ", "It is about %s. "]
    elif task=="subj":
        templates = ["This is %s . ", "It's all %s . ",
                     "It's %s . ", "Is it %s ? "]
    elif task=="CoLA":
        templates = ["This is %s .",
                     "It is %s .",
                     "You are %s .",
                     "I am %s ."]
    elif task in ['qnli']:
        templates = ['Given the premise, the question is %s.']
    elif task in ['wnli','snli','mnli']:
        templates = ['Given the premise, the hypothesis is %s.']
    elif task in ['atis', 'liu', 'facebook', 'snips']:
        templates = ['Intent: %s.']
    elif task in ['20News', 'NewsCategory']:
        templates = ['Category: %s.']
    else:
        raise NotImplementedError(task)

    if task in ['qnli']:
        label_words = ['unanswerable', 'answerable']
    elif task in ['wnli']:
        label_words = ['false or inconclusive', 'true']
    elif task in ['snli', 'mnli']:
        label_words = ['inconclusive','true','false']
    elif task in ["SST-2", "mr", "cr", "yelp_binary",'amazon_b']:
        label_words = ["terrible", "great"]
    elif task in ["sst-5", "yelp_full", "amazon"]:
        label_words = ["terrible", "bad", "okay", "good", "great"]
    elif task in ["agnews"]:
        label_words = ["World", "Sports", "Business", "Technology"]
    elif task in ["trec"]:
        label_words = ["Description", "Entity", "Expression",
                       "Human", "Location", "Number"]
    elif task in ["sogou"]:
        label_words = ["Sports", "Finance", "Entertainment",
                       "Automobile", "Technology"]
    elif task in ["subj"]:
        label_words = ["subjective", "objective"]
    elif task in ["CoLA"]:
        label_words = ["not grammatical", "grammatical"]
    elif task in ["dbpedia"]:
        label_words = ["Company",
                       "Educational Institution",
                       "Artist",
                       "Athlete",
                       "Office Holder",
                       "Mean of Transportation",
                       "Building",
                       "Natural Place",
                       "Village",
                       "Animal",
                       "Plant",
                       "Album",
                       "Film",
                       "Written Work"]
    elif task in ["yahoo"]:
        label_words = ["Society & Culture",
                       "Science & Mathematics",
                       "Health",
                       "Education & Reference",
                       "Computers & Internet",
                       "Sports",
                       "Business & Finance",
                       "Entertainment & Music",
                       "Family & Relationships",
                       "Politics & Government"]

    elif task in ['20News']:
        label_words = [
                'IBM', 
                'Middle East Politics',
                'Windows XP', 
                'Motorcycles', 
                'Medicine', 
                'For Sale',
                'Religion', 
                'MS Windows',
                'Baseball', 
                'Auto', 
                'Hockey',
                'Mac', 
                'Graphics', 
                'Christianity',
                'Guns', 
                'Electronics', 
                'Space', 
                'Crypto',
                'Atheism', 
                'Politics', 
                'None'

        ]
    elif task in ['NewsCategory']:
        label_words = [
                'Politics', 
                'World News', 
                'Parenting', 
                'Money', 
                'Wellness',
                'Business', 
                'Weddings', 
                'Entertainment', 
                'Impact', 
                'Black Voices',
                'Queer Voices', 
                'Crime', 
                'Divorce', 
                'Food and Drink', 
                'Worldpost',
                'Parents', 
                'Travel', 
                'The Worldpost', 
                'Healthy Living', 
                'Taste',
                'Media', 
                'Culture and Arts', 
                'Style', 
                'Weird News', 
                'Style and Beauty',
                'Comedy', 
                'Home and Living', 
                'Sports', 
                'Environment', 
                'Education',
                'Good News', 
                'Fifty', 
                'Women', 
                'Science', 
                'Arts and Culture',
                'US News', 
                'Green', 
                'Tech', 
                'Religion', 
                'Latino Voices',
                'College', 
                'Arts'

        ]
    elif task in ['atis']:
        label_words = [
        'Flight',
            'Abbreviation',
            'City',
            'Airfare',
            'Ground Service',
            'Ground Fare',
            'Airline',
            'Flight Number',
            'Aircraft',
            'Distance',
            'Capacity',
            'Flight Time',
            'Quantity',
            'Airport',
        ]
    elif task in ['liu']:
        label_words = [
                 'Set Alarm',
                 'Definition',
                 'Play Music',
                 'Set Calendar',
                 'Play Radio',
                 'Confirm',
                 'Quirky',
                 'Send Email',
                 'Currency',
                 'Like',
                 'Social Query',
                 'News',
                 'Neutral',
                 'Stop',
                 'Calendar Query',
                 'Alarm Query',
                 'Recommend Movie',
                 'Play Audiobook',
                 'Weather Query',
                 'Praise',
                 'Social Post',
                 'Play Podcast',
                 'Negate',
                 'Recipe',
                 'Affirm',
                 'Remove Calendar',
                 'Recommend Location',
                 'Remove List',
                 'Explain',
                 'Email Query',
                 'Maths',
                 'Stock',
                 'Transport Query',
                 'Order Takeaway',
                 'Datetime Query',
                 'Repeat',
                 'Factoid',
                 'Taxi',
                 'Add List',
                 'Add Contact',
                 'Recommend Event',
                 'Mute Volume',
                 'List Query',
                 'Ticket',
                 'Convert Datetime',
                 'Joke',
                 'Remove Alarm',
                 'Traffic',
                 'Volume Up',
                 'Takeaway Query',
                 'Play Game',
                 'Contact Query',
                 'Music Query',
                 'Volume Other',
                 'Volume Down',
                 'Music Setting',
                 'Greet',
                 'Dislike',

        ]
    elif task in ['facebook']:
        label_words = [
            'Road Condition',
                'Departure',
                'Duration',
                'Event',
                'Arrival',
                'Directions',
                'Distance',
                'Traffic',
                'Update',
                'Combine',
                'Route',
                'Location',
        ]
    elif task in ['snips']:
        label_words = [
            'Playlist',
            'Weather',
            'Event',
            'Musing',
            'Creative Work',
            'Rate Book',
            'Book Restaurant',

        ]
    else:
        raise NotImplementedError(task)

    return [templates[idx] % label_word for label_word in label_words]


def get_paths(out_dir, gpt2, method, task, do_zeroshot,
              k, seed, train_seed, split, template_idx,
              batch_size=None, lr=None, warmup_steps=None,
              use_demonstrations=False,
              ensemble=False,
              prompt_tune=False,
              head_tune=False,
              transform_tune=False,
              n_prefix=20):

    model_name = gpt2

    if not do_zeroshot:
        if prompt_tune:
            model_name += "-prompt-ft"
            if n_prefix!=20:
                model_name += "-{}".format(n_prefix)
        elif head_tune:
            model_name += "-head-ft"
        elif transform_tune:
            model_name += "-transform-ft"
        else:
            model_name += "-all-ft"

    base_dir = os.path.join(out_dir,
                            model_name,
                            "{}{}{}".format(method,
                                              "-demon" if use_demonstrations else "",
                                              "-ensemble" if ensemble else ""),
                            task)


    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if do_zeroshot:
        cache_path = str(split)

        if use_demonstrations:
            cache_path += "-k={}-seed={}".format(k, seed)

        if use_demonstrations:
            cache_path += "-tseed={}".format(train_seed)

        cache_path += "-t={}".format(template_idx)

        return os.path.join(base_dir, cache_path+".pkl")

    assert batch_size is not None and lr is not None and warmup_steps is not None

    out_dir = "BS={}-k={}-t={}-seed={}-tseed={}-lr={}{}".format(
            batch_size, k, template_idx, seed, train_seed, lr,
            "-wamrup={}".format(warmup_steps) if warmup_steps>0 else "",
    )


    return os.path.join(base_dir, out_dir)


def prepend_task_tokens(tokenizer, inputs, n_prefix):
    task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(n_prefix)]
    tokenizer.add_tokens(task_tokens)
    task_token_ids = tokenizer(" ".join(task_tokens), return_tensors="pt")["input_ids"]
    assert task_token_ids.shape[-1]==n_prefix

    def convert(inputs):
        n_train = inputs["input_ids"].shape[0]

        new_input_ids=torch.cat([
                task_token_ids.repeat(n_train, 1),
                inputs["input_ids"][:,1:]], 1)

        inputs = dict(
            input_ids=new_input_ids,
            attention_mask=torch.cat([
                torch.ones((n_train, n_prefix-1), dtype=torch.long),
                inputs["attention_mask"]], 1),
            token_type_ids=torch.cat([
                torch.zeros((n_train, n_prefix-1), dtype=torch.long),
                inputs["token_type_ids"]], 1),
            labels=torch.cat([
                torch.zeros((n_train, n_prefix-1), dtype=torch.long),
                inputs["input_ids"]], 1))
        return inputs

    if type(inputs)==list:
        return [convert(_inputs) for _inputs in inputs]

    return convert(inputs)

def reassign_output_tokens(inputs, for_labels=True, mapping=None):
    '''
    if for_labels=True, keep input_ids and convert labels
    otherwise, keep labels and convert input_ids
    '''

    def get_unique_tokens(inputs):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach().numpy().tolist()
        unique_tokens = set()
        for _input_ids, _token_type_ids in zip(input_ids, token_type_ids):
            unique_tokens |= set([_id for _id, _token_id in zip(_input_ids, _token_type_ids) if _token_id==int(for_labels)])
        return unique_tokens

    def convert_set_to_mapping(unique_tokens):
        unique_tokens = sorted(unique_tokens)
        return {token: new_token for new_token, token in enumerate(unique_tokens)}

    def apply_mapping(inputs, mapping):
        input_ids = inputs["input_ids"].detach().numpy().tolist()
        token_type_ids = inputs["token_type_ids"].detach().numpy().tolist()
        converted_input_ids = []
        for _input_ids, _token_type_ids in zip(input_ids, token_type_ids):
            converted_input_ids.append([])
            for _id, _token_id in zip(_input_ids, _token_type_ids):
                if _token_id==int(for_labels):
                    converted_input_ids[-1].append(mapping[_id])
                else:
                    converted_input_ids[-1].append(0)
        converted_input_ids = torch.LongTensor(converted_input_ids)
        if for_labels:
            return dict(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        labels=converted_input_ids)
        return dict(input_ids=converted_input_ids,
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    labels=inputs["input_ids"])

    if type(inputs)==list:
        if mapping is None:
            unique_tokens = set()
            for _inputs in inputs:
                unique_tokens |= get_unique_tokens(_inputs)
            mapping = convert_set_to_mapping(unique_tokens)
        rev_mapping = {v:k for k, v in mapping.items()}
        return rev_mapping, [apply_mapping(_inputs, mapping) for _inputs in inputs]

    assert mapping is None
    mapping = convert_set_to_mapping(get_unique_tokens(inputs))
    rev_mapping = {v:k for k, v in mapping.items()}
    return rev_mapping, apply_mapping(inputs, mapping)




