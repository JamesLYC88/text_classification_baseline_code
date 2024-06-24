import argparse
import os
import re
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dl', '--data_list', type=list,
        default=['ecthr_a', 'ecthr_b', 'scotus', 'eurlex', 'ledgar', 'unfair_tos'])
    parser.add_argument('-sl', '--split_list', type=list, default = ['train', 'validation', 'test'])
    parser.add_argument('-ddp', '--data_dir_prefix', type=str, default='data')
    parser.add_argument('-f', '--format', type=str, choices=['linear', 'nn', 'hier'], required=True)
    args = parser.parse_args()
    return args


def data2task(data):
    return 'multi_label' if data not in ['scotus', 'ledgar'] else 'multi_class'


def data2hier(data):
    return True if data in ['ecthr_a', 'ecthr_b', 'scotus'] else False


def split2name(split):
    _split2name = {'train': 'train.txt', 'validation': 'valid.txt', 'test': 'test.txt'}
    return _split2name[split]


def get_texts(data, dataset, hier=False):
    if 'ecthr' in data:
        if not hier:
            texts = [' '.join(text) for text in dataset['text']]
        else:
            texts = [' [HIER] '.join(text) for text in dataset['text']]
        return [' '.join(text.split()) for text in texts]
    elif data == 'scotus' and hier:
        texts = [' [HIER] '.join(re.split('\n{2,}', text)) for text in dataset['text']]
        # Huggingface tokenizer ignores newline and tab,
        # so it's okay to replace them with a space here.
        for i in range(len(texts)):
            texts[i] = texts[i].replace('\n', ' ')
            texts[i] = texts[i].replace('\r', ' ')
            texts[i] = texts[i].replace('\t', ' ')
        return texts
    elif data == 'case_hold':
        return [contexts[0] + ' [SEP] '.join(holdings)
                for contexts, holdings in zip(dataset['contexts'], dataset['endings'])]
    else:
        return [' '.join(text.split()) for text in dataset['text']]


def get_labels(data, dataset, task):
    if task == 'multi_class':
        return list(map(str, dataset['label']))
    else:
        if data == 'eurlex':
            return [' '.join(map(str, [l for l in label if l < 100])) for label in dataset['labels']]
        else:
            return [' '.join(map(str, label)) for label in dataset['labels']]


def save_data(data_path, data):
    with open(data_path, 'w') as f:
        for text, label in zip(data['text'], data['labels']):
            assert '\n' not in label+text
            assert '\r' not in label+text
            assert '\t' not in label+text
            formatted_instance = '\t'.join([label, text])
            f.write(f'{formatted_instance}\n')


def main():
    # args
    args = get_args()
    data_dir = f'{args.data_dir_prefix}_{args.format}'
    os.makedirs(data_dir, exist_ok=True)

    # generate
    for data in args.data_list:
        if args.format == 'hier' and not data2hier(data):
            continue
        data_path = os.path.join(data_dir, data)
        os.makedirs(data_path, exist_ok=True)
        processed_data = {}
        for split in args.split_list:
            dataset = load_dataset('coastalcph/lex_glue', data, split=split, trust_remote_code=True)
            texts = get_texts(data, dataset, hier=args.format == 'hier')
            labels = get_labels(data, dataset, data2task(data))
            assert len(texts) == len(labels)
            print(f'{data} ({split}): num_instance = {len(texts)}')
            processed_data[split] = {'text': texts, 'labels': labels}
        # format
        if args.format == 'linear':
            # train
            train_path = os.path.join(data_path, split2name('train'))
            train_data = {
                'text': processed_data['train']['text'] + processed_data['validation']['text'],
                'labels': processed_data['train']['labels'] + processed_data['validation']['labels']
            }
            save_data(train_path, train_data)
            # test
            test_path = os.path.join(data_path, split2name('test'))
            test_data = processed_data['test']
            save_data(test_path, test_data)
        elif args.format == 'nn' or args.format == 'hier':
            # train/validation/test
            for split in processed_data:
                split_path = os.path.join(data_path, split2name(split))
                save_data(split_path, processed_data[split])


if __name__ == '__main__':
    main()
