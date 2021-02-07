import json
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertJapaneseTokenizer, BertConfig
from src.model import Predictor
from src.preprocess import preprocess
from src.constant import JUN, GYAKU, NEUTRAL

tweet_filename = './data/tweet.json'
dataset_filename = './sample.txt'


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


def splitter(x):
    tag, sent = x.split(',', 1)
    tag = int(tag)
    return (tag, sent)


def load_dataset(filename):
    with open(dataset_filename, 'r') as f:
        dataset = f.readlines()

    dataset = list(map(splitter, dataset))

    return dataset


def main():
    # omoshiro_tweets = load_omoshiro_tweets_from_json(tweet_filename)

    config_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    dataset = load_dataset(dataset_filename)

    tokenizer = BertJapaneseTokenizer.from_pretrained(config_path)
    config = BertConfig.from_pretrained(config_path)

    dataset = preprocess(dataset, tokenizer, config, batch_size=8)

    model = Predictor(config_path=config_path, model_path=config_path)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=NEUTRAL)

    optimizer = optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(5):
        for batch in dataset:
            model.zero_grad()

            src = batch['src']
            tgt = batch['tgt']

            # output = [batch_size, vocab_size]
            output = model(src)
            loss = criterion(output, tgt.squeeze())

            loss.backward()
            optimizer.step()

            print(loss)


if __name__ == '__main__':
    main()
