import torch
from transformers import BertJapaneseTokenizer
from .constant import JUN, GYAKU, NEUTRAL


def _preprocess_dataset(src_data, tokenizer):
    """
    dataset = [(ids, tag), ...]
    src = [id1, id2, ...]
    tgt: int
    """
    dataset = []
    max_length = 0
    for (tag, sent) in src_data:
        token_ids = tokenizer.encode(sent)
        dataset.append({'src': token_ids, 'tgt': tag})
        if max_length < len(token_ids):
            max_length = len(token_ids)

    return dataset, max_length


def _mk_batch(dataset, max_length, batch_size, device, pad):
    batch = _mk_initial_batch(max_length, batch_size, device, pad)

    for i, data in enumerate(dataset):
        apparent_length = min(len(data['src']), max_length)
        batch['src'][i][:apparent_length] = torch.Tensor(data['src'])
        batch['tgt'][i][0] = data['tgt']

    return batch


def _mk_initial_batch(max_length, batch_size, device, pad):
    batch = {}
    batch['src'] = torch.full((batch_size, max_length), fill_value=pad, device=device, dtype=torch.long)
    batch['tgt'] = torch.full((batch_size, 1), fill_value=NEUTRAL, device=device, dtype=torch.long)

    return batch


def _mk_batches(dataset, max_length, batch_size, device, pad):
    data_order = torch.randperm(len(dataset))
    dataset = [dataset[idx] for idx in data_order]
    batches = []

    current_data = []

    for i, d in enumerate(dataset):
        if i % batch_size == 0 and current_data != []:
            batch = _mk_batch(current_data, max_length, batch_size, device, pad)
            batches.append(batch)
            current_data = []

        current_data.append(d)

    return batches


def preprocess(src_data, tokenizer, config, batch_size):
    pad_token_id = config.pad_token_id

    dataset, max_length = _preprocess_dataset(src_data, tokenizer)

    print('max_length is {}'.format(max_length))

    dataset = _mk_batches(dataset=dataset, max_length=max_length, batch_size=batch_size, device='cpu', pad=pad_token_id)

    return dataset