import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertJapaneseTokenizer, BertConfig
from src.model import BertPredictor, Perceptron
from src.preprocess import preprocess, mk_batches
from src.constant import JUN, GYAKU, NEUTRAL

tweet_filename = './data/tweet.json'
dataset_filename = './data/sample.txt'


def load_omoshiro_tweets_from_json(filename):
    with open(tweet_filename, 'r') as f:
        tweets = json.load(f)

    omoshiro_tweets = []

    print('All tweets size : {}'.format(len(tweets)))

    for tweet in tweets:
        tweet_text = tweet['tweet']['full_text']

        # RT を除去する
        if 'RT @' in tweet_text:
            continue

        if '面白い' in tweet_text or '面白すぎ' in tweet_text:
            omoshiro_tweets.append(tweet_text.replace('\n', ''))
            # print(omoshiro_tweets[-1])

    return omoshiro_tweets



def load_dataset(filename):
    def _splitter(x):
        tag, sent = x.split(',', 1)
        tag = int(tag)
        return (tag, sent)

    with open(dataset_filename, 'r') as f:
        dataset = f.readlines()

    dataset = list(map(_splitter, dataset))

    return dataset


def main():
    # omoshiro_tweets = load_omoshiro_tweets_from_json(tweet_filename)
    valid_dataset_size = 128
    batch_size = 8
    config_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = BertJapaneseTokenizer.from_pretrained(config_path)
    config = BertConfig.from_pretrained(config_path)

    pad_token_id = config.pad_token_id

    dataset = load_dataset(dataset_filename)

    train_dataset = dataset[:-128]
    valid_dataset = dataset[-128:]

    train_dataset, train_max_length = preprocess(train_dataset, tokenizer, config, batch_size=batch_size, device=device)
    valid_dataset, valid_max_length = preprocess(valid_dataset, tokenizer, config, batch_size=batch_size, device=device)

    valid_batches = mk_batches(dataset=valid_dataset, max_length=valid_max_length, batch_size=batch_size, device=device, pad=pad_token_id)

    print('Train dataset size is {}, Valid dataset size is {}'.format(len(train_dataset), len(valid_dataset)))

    # model = BertPredictor(config_path=config_path, model_path=config_path)
    model = Perceptron(vocab_size=tokenizer.vocab_size, hidden_size=128, device=device)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=NEUTRAL)

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(40):
        print('------ Epoch {} ------'.format(epoch + 1))

        train_batches = mk_batches(dataset=train_dataset, max_length=train_max_length, batch_size=batch_size, device=device, pad=pad_token_id)

        print('Train')
        model.train()
        accuracy = 0.0
        for batch in train_batches:
            model.zero_grad()

            src = batch['src']
            tgt = batch['tgt']

            # output = [batch_size, vocab_size]
            output = model(src)

            loss = criterion(output, tgt)

            labels = torch.argmax(output, dim=-1)

            accuracy += (labels == tgt).sum()

            loss.backward()
            optimizer.step()

            sys.stdout.write('\rLoss {}                '.format(loss.item()))
        
        accuracy /= len(train_dataset)

        print('\nTrain accuracy {}'.format(accuracy))

        print('Validation')
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            for batch in valid_batches:
                src = batch['src']
                tgt = batch['tgt']

                output = model(src)

                labels = torch.argmax(output, dim=-1)

                accuracy += (labels == tgt).sum()
            
            accuracy /= valid_dataset_size
            print('Valid accuracy : {}'.format(accuracy))

    accuracy = 0.0
    for batch in valid_batches:
        accuracy += (JUN == batch['tgt']).sum()
    
    accuracy /= valid_dataset_size

    print('== JUN accuracy : {}'.format(accuracy))

if __name__ == '__main__':
    main()
