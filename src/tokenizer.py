#import pygtrie
import regex as re
from collections.abc import Iterable, Iterator 
import os
from typing import BinaryIO

import numpy as np
import regex as re
import mmap
from multiprocessing import Pool
from collections import  Counter , defaultdict


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def init_worker_mmap(input_path):
    global _mm, _input_path
    _input_path = input_path
    f = open(_input_path, 'rb')
    _mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # Don't close `f` — mmap needs it open!


def pre_tokenization(
    start: int,
    end: int,
    pattern: re.Pattern,
    special_tokens_splits: str
) -> Counter[str, int] :
    """
    Given a chunck and a pre-tokenization regex startegy pattern and split_pattern for special tokens
    return the pre_tokens generated (not in tuple)
    """
    chunk = _mm[start:end].decode("utf-8", errors="ignore")
    splits = re.split(special_tokens_splits, chunk)

    pre_tokens = Counter()
    
    for split in splits:
        pre_tokens.update(Counter(tuple(bytes(match.group(0).encode("utf-8"))) for match in pattern.finditer(split)))

    return pre_tokens

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Intialize the vocabulary
    vocab = {i: bytes([i]) for i in np.arange(256, dtype=np.uint8)}

    for idx ,special_token in enumerate(special_tokens, start = 256) :
        vocab[idx] = special_token.encode("utf-8")
    # Pre tokenization
    num_processes = 16
    with open(input_path,'rb') as f :
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)  
    sp_tokens_split = re.escape('|'.join(special_tokens))
    args = [(start, end, pattern, sp_tokens_split) for start, end in zip(boundaries[:-1], boundaries[1:])]

    
    with Pool(num_processes, initializer=init_worker_mmap, initargs=(input_path,)) as p :
        results = p.starmap(pre_tokenization, args) 
    
    pre_tokens = dict(sum(results, Counter()))
    

    #Merging 
    #first step
    pair_freq = Counter()
    pair_occurrences = defaultdict(list)
    pre_token_w_update ={}

    word_idx = 0
    for key in pre_tokens :
        for i in range(len(key) -1) :
            pair = (key[i] , key[i+1])
            pair_freq[pair] += pre_tokens[key]  
            pair_occurrences[pair].append(word_idx)
        pre_token_w_update[word_idx] = (list(key), pre_tokens[key]) # list so that if a pair is present multiple time but 'disapear' because of a neihhbor merging we dn't forget the remaing pair
        word_idx += 1
    # heap = [(freq, pair) for pair, freq in pair_freq.items()]
    # heapq._heapify_max(heap)

    
    merges = []
    vocab_merged_id = {}

    n_merge = vocab_size - len(vocab)

    for i in range(n_merge): 
    #     while True:
    #         freq, pair = heapq._heappop_max(heap)
    #         if pair in pair_freq and freq == pair_freq[pair]:
    #             break

        pair = max(pair_freq.items(), key = lambda x: (x[1], (vocab[x[0][0]],vocab[x[0][1]])))[0]


        merges.append((bytes(vocab[pair[0]]), bytes(vocab[pair[1]])))
        id = len(vocab)
        vocab[id] = vocab[pair[0]] + vocab[pair[1]]
        pair_freq[pair] = 0 # remove pair from counter entirely ?


        # updated_pair = set()
        for idx in set(pair_occurrences[pair]) : #if pair is in word multiple time we nly need to go through word once 
            #find where the pair occuredin the word
            i = 0
            word = pre_token_w_update[idx][0]
            while(i < len(word)-1):
                #look where does the pair occur in the word
                if pair[0] == word[i] and pair[1] == word[i+1] :
                    if i>0 :
                        left = word[i-1]
                        left_pair = (left, pair[0])

                        pair_occurrences[left_pair].remove(idx)
                        pair_freq[left_pair] -= pre_token_w_update[idx][1]
                        # updated_pair.add(left_pair)

                        pair_freq[(left,id)] += pre_token_w_update[idx][1]
                        pair_occurrences[(left,id)].append(idx)
                        # updated_pair.add((left,id))
                    #some nasty code duplication to refractor later
                    if i + 2 <len(word) :
                        right = word[i+2]
                        right_pair = (pair[1], right)
                        

                        pair_occurrences[right_pair].remove(idx)
                        pair_freq[right_pair] -= pre_token_w_update[idx][1]
                        # updated_pair.add(right_pair)

                        pair_freq[(id,right)] += pre_token_w_update[idx][1]
                        pair_occurrences[(id,right)].append(idx)
                        # updated_pair.add((id,right))
                    
                    word[i:i+2] = [id]
                    #do the updates
                i +=1 #
        pair_occurrences[pair] = []# remove pair from dict entirely ?


        
        # remove pair from pair freq & pair occurence 
    



    return vocab, merges



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer: 

    def __init__(self, vocab, merges, special_tokens=None) :
        '''
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        the following parameters:

        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None : 
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else : 
            self.special_tokens = None

        self.vocab_to_id = {v : k for k ,v in vocab.items()}
        self.pattern = re.compile(PAT) 
        #self.vocab_trie = self.build_vocab_trie(vocab)


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        raise NotImplementedError 

    def encode(self, text: str) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        splits = self._split_special_token(text)
        result = []
        for split in splits : 
            if self.special_tokens is not None and split in self.special_tokens :
                result.append(self.vocab_to_id[split.encode("utf-8")])
            else :
                words = self._pretokenize(split)
                for word in words:
                    for merge in self.merges:
                        i = 0
                        while(i < len(word)-1):
                            if merge[0] == word[i] and merge[1] == word[i+1] :
                                word[i:i+2] = [merge[0] + merge[1]]
                            i+=1
                    for token in word:
                        1 == 1 
                        result.append(self.vocab_to_id[bytes(token)])
        return result



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        '''
        for text in iterable:
            for id in self.encode(text):
                yield id


    def decode(self, ids: list[int]) -> str :
        '''Decode a sequence of token IDs into text. '''
        bytes = b''
        for id in ids :
            bytes += self.vocab[id]
        return bytes.decode('utf-8', errors='replace' )


    # @staticmethod
    # def build_vocab_trie(vocab: dict[int, bytes]):
    #     trie = pygtrie.Trie()
    #     for token_id, byte_seq in vocab.items():
    #         key = tuple(byte_seq)
    #         trie[key] = token_id
    #     return trie
    

    def _pretokenize(self, split):
        return list(list(bytes([b]) for b in match.group(0).encode("utf-8")) for match in self.pattern.finditer(split))
    

    def _split_special_token(self, text: str)-> list[str] :
        if self.special_tokens is None:
            return [text]
        else :
            sp_tokens_pattern = '|'.join(map(re.escape, self.special_tokens))
    
            # Use re.split with a capturing group to keep the separators
            splits = re.split(f'({sp_tokens_pattern})', text)

            return splits
        
    

