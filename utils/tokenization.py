from collections import Counter
import bpemb
from collections import Counter
import numpy as np
from tqdm import tqdm_notebook

class Vocab:
    def __init__(self, word_counts, size, specials, unk_index):
        self._word_list = []
        self._word_list.extend(specials)
        for w,_ in word_counts.most_common(size-len(specials)):
            self._word_list.append(w)
        self._reverse_index = {w:i for i,w in enumerate(self._word_list)}
        self.unk_index = unk_index
        self.n_specials = len(specials)
        
    @classmethod
    def from_id2word(cls,id2word, unk_index, n_specials):
        self = cls.__new__(cls)
        self._word_list = [x for x in id2word]
        self._reverse_index = {w:i for i,w in enumerate(self._word_list)}
        self.unk_index = unk_index
        self.n_specials = n_specials
        return self
        
    def word2id(self, w):
        idx = self._reverse_index.get(w)
        if idx is not None:
            return idx
        else:
            return self.unk_index
        
    def id2word(self, idx):
        return self._word_list[idx]
    
    def transform_tokens(self, text, drop_unk=False):
        result = []
        for tok in text:
            idx = self.word2id(tok)
            if idx == self.unk_index and drop_unk:
                continue
            result.append(idx)
        return result
    
    def transform_ids(self, ids):
        return [self.id2word(x) for x in ids]
            
    def __len__(self):
        return len(self._word_list)
    
def encode_text(bpe, text, append_left, append_right):
    unified_whitespace_text = ' '.join(text.split())
    bpe =  bpe.encode(unified_whitespace_text)
    if append_left:
        bpe = [append_left] + bpe
    if append_right:
        bpe.append(append_right)
    return bpe

def encode_texts(bpe, texts, append_left=None, append_right=None):
    result = []
    for text in tqdm_notebook(texts):
        result.append(encode_text(bpe, text, append_left, append_right))
    return result
    
def build_vocab_from_pretrained_bpe(texts, special_symbols, bpe, max_size, unk_index):
    """Counts subwords in text, builds vocab limited by max_size (throws out the least frequent subwords)
    Doesn't include any subwords, which don't have embedding
    """
    oov_subwords = Counter()
    iv_subwords = Counter()
    for tokens in tqdm_notebook(texts):
        for sw in tokens:
            if sw not in bpe.emb:
                oov_subwords[sw] += 1
            else:
                iv_subwords[sw] += 1 
                
    print('OOV subwords: ', oov_subwords.most_common(10))
    vocab = Vocab(iv_subwords, max_size, special_symbols, unk_index)
    return vocab

def extract_pretrained_bpe_embeddings(vocab, bpe, init_unk='mean'):
    embeddings = np.zeros(shape=(len(vocab), bpe.emb.vector_size), dtype=np.float32)
    for i in tqdm_notebook(range(vocab.n_specials, len(vocab))):
        w = vocab.id2word(i)
        assert w in bpe.emb
        embeddings[i] = bpe.emb[w]
    if init_unk == 'mean':
        embeddings[vocab.unk_index] = np.mean(embeddings[vocab.n_specials:], axis=0)
    elif init_unk is not None:
        raise ValueError('init_unk should be in (\'mean\', None)')
    return embeddings

def transform_bpe_to_ids(bpe_encoded_texts, vocab):
    return [vocab.transform_tokens(text) for text in bpe_encoded_texts]