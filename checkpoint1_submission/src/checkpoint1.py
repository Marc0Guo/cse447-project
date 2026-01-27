#!/usr/bin/env python
import os
import sys
import string
import random
import pickle
import collections
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self, n=4):
        self.n = n
        # counts[context][next_char] = frequency
        self.ngram_counts = collections.defaultdict(collections.Counter)
        self.unigram_counts = collections.Counter()
        self.fallback_chars = [' ', 'e', 'a', 't', 'i', 'n', 'o', 's']

    @classmethod
    def load_training_data(cls):
        # Load from example/input.txt
        data = []
        input_file = '../../example/input.txt'
        if os.path.exists(input_file):
            with open(input_file) as f:
                for line in f:
                    data.append(line.strip())
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        print(f"Training on {len(data)} lines of text...")
        for line in data:
            for char in line:
                self.unigram_counts[char] += 1

            for i in range(len(line) - 1):
                for j in range(1, self.n):
                    if i - j + 1 >= 0:
                        context = line[i - j + 1:i + 1]
                        next_char = line[i + 1]
                        self.ngram_counts[context][next_char] += 1

    def _get_top_candidates(self, history):
        candidates =[]
        seen_candidates = set()

        def add_candidates(counter):
            for char, _ in counter.most_common():
                if char not in seen_candidates:
                    candidates.append(char)
                    seen_candidates.add(char)
                    if len(candidates) >= 3:
                        return True
            return False
        
        for i in range(self.n - 1, 0, -1):
            if len(history) >= i:
                context = history[-i:]
                if context in self.ngram_counts:
                    if add_candidates(self.ngram_counts[context]):
                        return candidates[:3]
        
        if add_candidates(self.unigram_counts):
            return candidates[:3]
        
        for char in self.fallback_chars:
            if char not in seen_candidates:
                candidates.append(char)
                seen_candidates.add(char)
                if len(candidates) >= 3:
                    break
                    
        return candidates[:3]

    def run_pred(self, data):
        # your code here
        preds = []
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = self._get_top_candidates(inp)
            preds.append(''.join(top_guesses))
        return preds
    
    def run_interactive(self):
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        
        history = ""
        while True:
            top_3 = self._get_top_candidates(history)
            print(f"{top_3[0]}{top_3[1]}{top_3[2]}", flush=True)

            try:
                next_char = sys.stdin.read(1)
            except (IOError, EOFError):
                break
            
            if not next_char:
                break
            
            self.unigram_counts[next_char] += 1
            for j in range(1, self.n):
                if len(history) >= j:
                    context = history[-j:]
                    self.ngram_counts[context][next_char] += 1
            
            history += next_char
            if len(history) > 100:
                history = history[-self.n:]

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        model_path = os.path.join(work_dir, 'model.checkpoint')
        print(f'Saving model to {model_path}')

        with open(model_path, 'wb') as f:
            pickle.dump({
                'ngram': self.ngram_counts,
                'unigram': self.unigram_counts,
                'n': self.n
            }, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model_path = os.path.join(work_dir, 'model.checkpoint')
        print(f"Loading model from {model_path}...")

        model = MyModel()
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                model.ngram_counts = saved_data['ngram']
                model.unigram_counts = saved_data['unigram']
                model.n = saved_data['n']
        else:
            print("Warning: No checkpoint found, using empty model.")
            
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
