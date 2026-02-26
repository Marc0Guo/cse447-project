#!/usr/bin/env python
import os
import sys
import string
import random
import pickle
import collections
import re
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm


class MyModel:
    """
    N-gram model with backoff for character prediction.
    Uses direct count lookup for fast O(1) prediction per context length.
    """
    _ws_re = re.compile(r'\s+')

    def __init__(self, n=6):
        self.n = n
        # ngram_counts[context_string] = Counter({next_char: count})
        # Contexts of lengths 1..n-1 are all stored here
        self.ngram_counts = collections.defaultdict(collections.Counter)
        self.unigram_counts = collections.Counter()
        self.vocab = set()
        self.fallback_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'r', 'h']
        # Pre-computed top-3 predictions per context (built after training/loading)
        self._top3 = {}
        self._unigram_top3 = []

    @staticmethod
    def _normalize(text):
        """Normalize: lowercase and collapse whitespace."""
        text = text.lower()
        text = MyModel._ws_re.sub(' ', text)
        return text

    def _normalize_tail(self, text, length):
        """Normalize only the last `length` characters of text. Much faster for long inputs."""
        tail = text[-length:] if len(text) > length else text
        return tail.lower() if ' ' not in tail and '\t' not in tail and '\n' not in tail \
            else MyModel._ws_re.sub(' ', tail.lower())

    def _build_top3_cache(self):
        """Pre-compute top-3 next chars for every stored context. Call after training or loading."""
        self._top3 = {}
        for ctx, counter in self.ngram_counts.items():
            self._top3[ctx] = [ch for ch, _ in counter.most_common(3)]
        self._unigram_top3 = [ch for ch, _ in self.unigram_counts.most_common(3)]

    @classmethod
    def load_training_data(cls):
        """Load training data from Wikitext dataset (requires: pip install datasets)."""
        from datasets import load_dataset

        print("Loading Wikitext training dataset from HuggingFace...")
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')

        data = []
        for item in dataset:
            text = item['text'].strip()
            if text:
                data.append(text)

        print(f"Loaded {len(data)} lines from Wikitext training set")
        return data

    @classmethod
    def load_test_data(cls, fpath=None, split='test'):
        if fpath:
            data = []
            with open(fpath, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    data.append(line.rstrip('\n\r'))
            return data

        from datasets import load_dataset
        print(f"Loading Wikitext {split} split from HuggingFace...")
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split=split)
        data = []
        for item in dataset:
            text = item['text'].strip()
            if text and len(text) > 1:
                data.append(text[:-1])
        print(f"Loaded {len(data)} test samples from Wikitext {split} split")
        return data

    @classmethod
    def load_test_answers(cls, split='test'):
        from datasets import load_dataset
        print(f"Loading Wikitext {split} answers from HuggingFace...")
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split=split)
        answers = []
        for item in dataset:
            text = item['text'].strip()
            if text and len(text) > 1:
                answers.append(text[-1])
        print(f"Loaded {len(answers)} answers from Wikitext {split} split")
        return answers

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8', newline='\n') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        total = len(data)
        print(f"Training on {total} lines...")
        t_start = time.perf_counter()
        total_chars = 0

        for line in tqdm(data, desc="Training", unit="line"):
            text = self._normalize(line)
            if not text:
                continue

            total_chars += len(text)

            # Collect unigram counts
            for ch in text:
                self.unigram_counts[ch] += 1
                self.vocab.add(ch)

            # Collect n-gram counts for all context lengths
            for i in range(len(text)):
                ch = text[i]
                max_ctx = min(self.n - 1, i)
                for ctx_len in range(1, max_ctx + 1):
                    ctx = text[i - ctx_len:i]
                    self.ngram_counts[ctx][ch] += 1

        elapsed = time.perf_counter() - t_start
        print(f"Training done in {elapsed:.1f}s ({total_chars:,} chars, {total_chars/elapsed:,.0f} chars/s)")
        print(f"Vocab size: {len(self.vocab)}")
        print(f"Unique contexts: {len(self.ngram_counts)}")
        print(f"Total chars: {total_chars:,}")

        t_cache = time.perf_counter()
        print("Building top-3 prediction cache...")
        self._build_top3_cache()
        print(f"Cache built in {time.perf_counter() - t_cache:.2f}s")

    def _get_top_candidates(self, history):
        """
        Fast top-3 prediction using pre-computed cache with backoff.
        """
        top3 = self._top3
        n = self.n
        # Only normalize the tail we actually need, not the entire input
        context = self._normalize_tail(history, n - 1)
        ctx_max = min(n - 1, len(context))

        # Fast path: try longest context first — if it has 3 candidates, return immediately
        for ctx_len in range(ctx_max, 0, -1):
            ctx = context[-ctx_len:]
            cached = top3.get(ctx)
            if cached:
                if len(cached) >= 3:
                    return cached
                break  # found a match but < 3 candidates, go to slow path

        # Slow path: merge across backoff levels (rare)
        seen = set()
        out = []
        for ctx_len in range(ctx_max, 0, -1):
            ctx = context[-ctx_len:]
            cached = top3.get(ctx)
            if not cached:
                continue
            for ch in cached:
                if ch not in seen:
                    out.append(ch)
                    seen.add(ch)
                if len(out) == 3:
                    return out

        # Fallback to pre-computed unigram top-3
        for ch in self._unigram_top3:
            if ch not in seen:
                out.append(ch)
                seen.add(ch)
            if len(out) == 3:
                return out

        # Final fallback
        for ch in self.fallback_chars:
            if ch not in seen:
                out.append(ch)
                seen.add(ch)
            if len(out) == 3:
                return out

        while len(out) < 3:
            out.append('e')
        return out

    def run_pred(self, data):
        preds = []
        get = self._get_top_candidates
        for inp in data:
            preds.append(''.join(get(inp)))
        return preds

    @classmethod
    def evaluate(cls, preds, answers, verbose=False):
        if len(preds) != len(answers):
            print(f"Warning: {len(preds)} predictions but {len(answers)} answers")

        correct = 0
        total = min(len(preds), len(answers))

        for i in range(total):
            pred_chars = preds[i]
            answer = answers[i].lower()

            if answer in pred_chars.lower():
                correct += 1
                if verbose and i < 10:
                    print(f"  Correct: '{pred_chars}' | Answer: '{answer}'")
            else:
                if verbose and i < 10:
                    print(f"  Wrong:   '{pred_chars}' | Answer: '{answer}'")

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*50}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        return accuracy

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

            # Online learning: update counts with normalized text
            norm_char = self._normalize(next_char)
            if norm_char:
                norm_history = self._normalize(history)
                self.unigram_counts[norm_char] += 1
                for j in range(1, self.n):
                    if len(norm_history) >= j:
                        ctx = norm_history[-j:]
                        self.ngram_counts[ctx][norm_char] += 1

            history += next_char
            if len(history) > 1000:
                history = history[-1000:]

    def save(self, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        print(f'Saving model to {path}')

        # Convert to regular dicts for efficient serialization
        ngram_dict = {k: dict(v) for k, v in self.ngram_counts.items()}

        with open(path, 'wb') as f:
            pickle.dump({
                'ngram_counts': ngram_dict,
                'unigram_counts': dict(self.unigram_counts),
                'n': self.n,
                'vocab': sorted(self.vocab),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        size = os.path.getsize(path) / (1024 * 1024)
        print(f"Model saved ({size:.2f} MB)")

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        print(f"Loading model from {path}...")

        model = MyModel()

        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)

            model.n = data.get('n', 6)

            # Load ngram_counts (handles both old and new format)
            raw_ngram = data.get('ngram_counts', {})
            model.ngram_counts = collections.defaultdict(collections.Counter)
            for ctx, counts in raw_ngram.items():
                if isinstance(counts, dict):
                    model.ngram_counts[ctx] = collections.Counter(counts)
                else:
                    model.ngram_counts[ctx] = counts

            # Load unigram counts
            raw_unigram = data.get('unigram_counts', {})
            if isinstance(raw_unigram, dict):
                model.unigram_counts = collections.Counter(raw_unigram)
            else:
                model.unigram_counts = raw_unigram

            # Load vocab
            vocab_data = data.get('vocab', [])
            model.vocab = set(vocab_data) if isinstance(vocab_data, (list, set)) else set()

            print(f"Model loaded: n={model.n}, vocab={len(model.vocab)}, contexts={len(model.ngram_counts)}")
        else:
            print("Warning: No checkpoint found, using empty model.")

        print("Building top-3 prediction cache...")
        model._build_top3_cache()
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'evaluate', 'interactive'),
                       help='train | test | evaluate | interactive')
    parser.add_argument('--work_dir', help='directory to save/load model checkpoint', default='work')
    parser.add_argument('--test_data', help='path to local test data file')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--n', type=int, default=6, help='n-gram order')
    parser.add_argument('--split', choices=('test', 'validation'), default='test',
                       help='Wikitext split for test/evaluate mode')
    parser.add_argument('--max_samples', type=int, default=0, help='limit number of samples (0 for all)')
    parser.add_argument('--verbose', action='store_true', help='verbose output for evaluation')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Creating working directory {args.work_dir}')
            os.makedirs(args.work_dir)

        print(f'Initializing model (n={args.n})')
        model = MyModel(n=args.n)

        print('Loading Wikitext training data...')
        train_data = MyModel.load_training_data()

        print('Training model...')
        model.run_train(train_data, args.work_dir)

        print('Saving model checkpoint...')
        model.save(args.work_dir)

        print('Training complete!')

    elif args.mode == 'test':
        t0 = time.perf_counter()
        print('Loading model checkpoint...')
        model = MyModel.load(args.work_dir)
        t_load = time.perf_counter() - t0
        print(f'>>> Checkpoint load time: {t_load:.3f}s')

        # Load test data from file
        if args.test_data:
            print(f'Reading test data from {args.test_data}')
            test_data = MyModel.load_test_data(fpath=args.test_data)
        else:
            print(f'Loading Wikitext {args.split} split...')
            test_data = MyModel.load_test_data(split=args.split)

        if args.max_samples > 0 and len(test_data) > args.max_samples:
            print(f'Limiting to {args.max_samples} samples (from {len(test_data)})')
            test_data = test_data[:args.max_samples]

        n_samples = len(test_data)
        print(f'Making predictions on {n_samples} samples...')
        t1 = time.perf_counter()
        pred = model.run_pred(test_data)
        t_infer = time.perf_counter() - t1
        print(f'>>> Inference time: {t_infer:.3f}s ({n_samples} samples, {n_samples/t_infer:.0f} samples/s, {t_infer/n_samples*1000:.3f} ms/sample)')

        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)

        print('Testing complete!')

    elif args.mode == 'evaluate':
        t0 = time.perf_counter()
        print('Loading model checkpoint...')
        model = MyModel.load(args.work_dir)
        t_load = time.perf_counter() - t0
        print(f'>>> Checkpoint load time: {t_load:.3f}s')

        print(f'Loading Wikitext {args.split} split...')
        test_data = MyModel.load_test_data(split=args.split)
        answers = MyModel.load_test_answers(split=args.split)

        if args.max_samples > 0:
            print(f'Limiting to {args.max_samples} samples')
            test_data = test_data[:args.max_samples]
            answers = answers[:args.max_samples]

        n_samples = len(test_data)
        print(f'Evaluating on {n_samples} samples...')
        t1 = time.perf_counter()
        pred = model.run_pred(test_data)
        t_infer = time.perf_counter() - t1
        print(f'>>> Inference time: {t_infer:.3f}s ({n_samples} samples, {n_samples/t_infer:.0f} samples/s, {t_infer/n_samples*1000:.3f} ms/sample)')

        accuracy = MyModel.evaluate(pred, answers, verbose=args.verbose)
        print('Evaluation complete!')

    elif args.mode == 'interactive':
        print('Loading model checkpoint...')
        model = MyModel.load(args.work_dir)

        print('Starting interactive mode...')
        print('Type characters and the model will predict the next character.')
        print('Press Ctrl+C or Ctrl+D to exit.')
        model.run_interactive()

    else:
        raise NotImplementedError(f'Unknown mode: {args.mode}')
