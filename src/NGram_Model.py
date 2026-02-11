#!/usr/bin/env python
import os
import sys
import string
import random
import pickle
import collections
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class MyModel:
    """
    N-gram model with Kneser-Ney smoothing for character prediction.
    With backoff mechanism and vocabulary filtering.
    """
    def __init__(self, n=4, vocab_size=1000, discount=0.75):
        self.n = n
        self.vocab_size = vocab_size  # keep only top N characters
        self.discount = discount  # Kneser-Ney discount parameter

        # N-gram counts: ngram_counts[context][next_char] = count
        self.ngram_counts = collections.defaultdict(collections.Counter)

        # Unigram counts
        self.unigram_counts = collections.Counter()

        # Continuation counts for Kneser-Ney (how many unique contexts each char appears in)
        self.continuation_counts = collections.defaultdict(collections.Counter)

        # Total continuation count for normalization
        self.total_continuation = 0

        # Vocabulary (will be set after training)
        self.vocab = None

        # Fallback characters for extreme cases
        self.fallback_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h']

    @classmethod
    def load_training_data(cls, use_wikitext=True):
        """
        Load training data from Wikitext dataset or local file.
        For Wikitext: pip install datasets
        """
        data = []

        if use_wikitext:
            try:
                from datasets import load_dataset
                print("Loading Wikitext dataset from HuggingFace...")
                dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')

                # Process the dataset
                for item in dataset:
                    text = item['text'].strip()
                    if text and len(text) > 0:  # skip empty lines
                        data.append(text)

                print(f"Loaded {len(data)} lines from Wikitext")
                return data
            except Exception as e:
                print(f"Failed to load Wikitext: {e}")
                print("Falling back to local training data...")

        # Fallback to local file
        input_file = 'example/input.txt'
        if os.path.exists(input_file):
            with open(input_file, encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if text:
                        data.append(text)

        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8', newline='\n') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        print(f"Training on {len(data)} lines of text...")

        # First pass: collect all character counts
        all_char_counts = collections.Counter()
        for line in data:
            for char in line:
                all_char_counts[char] += 1

        # Build vocabulary: keep top N most frequent characters
        if self.vocab_size and self.vocab_size < len(all_char_counts):
            self.vocab = set([char for char, _ in all_char_counts.most_common(self.vocab_size)])
            print(f"Vocabulary size: {len(self.vocab)} (filtered from {len(all_char_counts)})")
        else:
            self.vocab = set(all_char_counts.keys())
            print(f"Vocabulary size: {len(self.vocab)}")

        # Second pass: collect n-gram counts (only for vocab characters)
        context_char_pairs = set()  # for continuation counts

        for line in data:
            # Filter line to only include vocab characters
            filtered_line = ''.join([c for c in line if c in self.vocab])

            if len(filtered_line) == 0:
                continue

            # Collect unigram counts
            for char in filtered_line:
                self.unigram_counts[char] += 1

            # Collect n-gram counts for all orders
            for i in range(len(filtered_line) - 1):
                next_char = filtered_line[i + 1]

                # Collect counts for different n-gram orders (1 to n-1)
                for order in range(1, self.n):
                    if i >= order - 1:
                        context = filtered_line[i - order + 1:i + 1]
                        self.ngram_counts[context][next_char] += 1

                        # Track context-char pairs for continuation counts
                        if order > 1:  # only for bigrams and higher
                            context_char_pairs.add((context, next_char))

        # Calculate continuation counts for Kneser-Ney
        # continuation_counts[context][char] = number of unique contexts where (context, char) appears
        for context, char in context_char_pairs:
            # For a given suffix, count how many unique prefixes it has
            if len(context) > 1:
                suffix = context[1:]
                self.continuation_counts[suffix][char] += 1

        # Calculate total continuation count (for unigram continuation probability)
        self.total_continuation = sum(sum(counts.values()) for counts in self.continuation_counts.values())

        print(f"Collected {len(self.ngram_counts)} unique contexts")
        print(f"Total unigrams: {sum(self.unigram_counts.values())}")
        print(f"Total continuation count: {self.total_continuation}")

    def _kneser_ney_prob(self, context, char):
        """
        Calculate Kneser-Ney smoothed probability for char given context.
        Uses recursive backoff with interpolation.
        """
        if not self.vocab or char not in self.vocab:
            return 1e-10  # very small probability for OOV characters

        # Base case: unigram (use continuation probability)
        if len(context) == 0:
            if self.total_continuation > 0:
                # Continuation probability: how many unique contexts this char appears in
                continuation_sum = sum(self.continuation_counts[ctx].get(char, 0)
                                      for ctx in self.continuation_counts)
                return max(continuation_sum / self.total_continuation, 1e-10)
            else:
                # Fallback to uniform unigram
                total = sum(self.unigram_counts.values())
                return self.unigram_counts.get(char, 1) / (total + len(self.vocab)) if total > 0 else 1e-10

        # Higher-order n-grams
        count_context_char = self.ngram_counts[context].get(char, 0)
        count_context = sum(self.ngram_counts[context].values())

        if count_context > 0:
            # Discounted probability
            discounted_prob = max(count_context_char - self.discount, 0.0) / count_context

            # Backoff weight (lambda)
            num_unique_continuations = len(self.ngram_counts[context])
            lambda_weight = (self.discount * num_unique_continuations) / count_context

            # Recursive backoff to lower-order model
            shorter_context = context[1:] if len(context) > 1 else ""
            backoff_prob = self._kneser_ney_prob(shorter_context, char)

            return discounted_prob + lambda_weight * backoff_prob
        else:
            # Pure backoff (no discounting needed)
            shorter_context = context[1:] if len(context) > 1 else ""
            return self._kneser_ney_prob(shorter_context, char)

    def _get_top_candidates(self, history):
        """
        Get top 3 character predictions using Kneser-Ney smoothing.
        Uses efficient hash table lookups.
        """
        # Filter history to only include vocab characters
        if self.vocab:
            filtered_history = ''.join([c for c in history if c in self.vocab])
        else:
            filtered_history = history

        # Use up to (n-1) characters as context
        max_context_len = self.n - 1
        context = filtered_history[-max_context_len:] if len(filtered_history) > 0 else ""

        # Get candidate characters (from vocab or all seen characters)
        candidates_set = self.vocab if self.vocab else set(self.unigram_counts.keys())

        # Calculate probabilities for all candidates
        char_probs = []
        for char in candidates_set:
            prob = self._kneser_ney_prob(context, char)
            char_probs.append((char, prob))

        # Sort by probability (descending)
        char_probs.sort(key=lambda x: x[1], reverse=True)

        # Get top 3
        top_3 = [char for char, _ in char_probs[:3]]

        # Fallback if we don't have enough candidates
        while len(top_3) < 3:
            for fallback_char in self.fallback_chars:
                if fallback_char not in top_3:
                    top_3.append(fallback_char)
                    break
            if len(top_3) >= 3 or len(top_3) >= len(self.fallback_chars):
                break

        return top_3[:3]

    def run_pred(self, data):
        preds = []
        for inp in data:
            top_guesses = self._get_top_candidates(inp)
            preds.append(''.join(top_guesses))
        return preds

    def run_interactive(self):
        """
        Interactive mode: continuously predict next character and learn from user input.
        """
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')

        history = ""
        while True:
            # Get top 3 predictions
            top_3 = self._get_top_candidates(history)
            print(f"{top_3[0]}{top_3[1]}{top_3[2]}", flush=True)

            # Read next character from user
            try:
                next_char = sys.stdin.read(1)
            except (IOError, EOFError):
                break

            if not next_char:
                break

            # Only update counts if character is in vocabulary (or no vocab filtering)
            if not self.vocab or next_char in self.vocab:
                self.unigram_counts[next_char] += 1

                # Update n-gram counts for all orders
                for j in range(1, self.n):
                    if len(history) >= j:
                        context = history[-j:]
                        self.ngram_counts[context][next_char] += 1

            # Update history (keep recent context)
            history += next_char
            if len(history) > 1000:  # keep last 1000 chars
                history = history[-1000:]

    def save(self, work_dir):
        """Save model checkpoint with all parameters and counts."""
        model_path = os.path.join(work_dir, 'model.checkpoint')
        print(f'Saving model to {model_path}')

        with open(model_path, 'wb') as f:
            pickle.dump({
                'ngram_counts': self.ngram_counts,
                'unigram_counts': self.unigram_counts,
                'continuation_counts': self.continuation_counts,
                'total_continuation': self.total_continuation,
                'n': self.n,
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'discount': self.discount
            }, f)

        # Print model statistics
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Model saved ({model_size:.2f} MB)")

    @classmethod
    def load(cls, work_dir):
        """Load model checkpoint from disk."""
        model_path = os.path.join(work_dir, 'model.checkpoint')
        print(f"Loading model from {model_path}...")

        model = MyModel()

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

                # Load all saved attributes
                model.ngram_counts = saved_data.get('ngram_counts', saved_data.get('ngram', collections.defaultdict(collections.Counter)))
                model.unigram_counts = saved_data.get('unigram_counts', saved_data.get('unigram', collections.Counter()))
                model.continuation_counts = saved_data.get('continuation_counts', collections.defaultdict(collections.Counter))
                model.total_continuation = saved_data.get('total_continuation', 0)
                model.n = saved_data.get('n', 4)
                model.vocab = saved_data.get('vocab', None)
                model.vocab_size = saved_data.get('vocab_size', 1000)
                model.discount = saved_data.get('discount', 0.75)

            print(f"Model loaded: n={model.n}, vocab_size={len(model.vocab) if model.vocab else 'unlimited'}")
        else:
            print("Warning: No checkpoint found, using empty model.")

        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'interactive'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--n', type=int, default=4, help='n-gram order (3 or 4 recommended)')
    parser.add_argument('--vocab_size', type=int, default=1000, help='vocabulary size (0 for unlimited)')
    parser.add_argument('--discount', type=float, default=0.75, help='Kneser-Ney discount parameter')
    parser.add_argument('--use_wikitext', action='store_true', help='use Wikitext dataset for training')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print(f'Instantiating model (n={args.n}, vocab_size={args.vocab_size}, discount={args.discount})')
        vocab_size = args.vocab_size if args.vocab_size > 0 else None
        model = MyModel(n=args.n, vocab_size=vocab_size, discount=args.discount)

        print('Loading training data')
        train_data = MyModel.load_training_data(use_wikitext=args.use_wikitext)

        if not train_data:
            print("ERROR: No training data loaded!")
            sys.exit(1)

        print('Training')
        model.run_train(train_data, args.work_dir)

        print('Saving model')
        model.save(args.work_dir)

        print('Training complete!')

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

        print('Testing complete!')

    elif args.mode == 'interactive':
        print('Loading model')
        model = MyModel.load(args.work_dir)

        print('Starting interactive mode...')
        model.run_interactive()
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
