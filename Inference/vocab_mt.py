# Includes class definitions for ASR and MT vocabularies

# Default tokens
PAD_token = 0  # Used for padding short number sequences
SOS_token = 1  # Start-of-sequence token
EOS_token = 2  # End-of-sequence token
UNK_token = 3  # Unknown token

class ASR_Vocab(object):
    def __init__(self, digit_seqs):
        super(ASR_Vocab, self).__init__()
        self.digit2index = {}
        self.digit2count = {}
        self.index2digit = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        
        # Count SOS, EOS, PAD, UNK
        self.num_tokens  = 4
        self.digit_seqs  = digit_seqs
        
    def dig2idx(self, digit):
        if digit in self.digit2index:
            return self.digit2index[digit]
        
    def idx2dig(self, idx):
        if idx in self.index2digit:
            return self.index2digit[idx]
            
    def add_digit(self, digit):
        if digit in self.digit2index:
            self.digit2count[digit] += 1
            
        else:
            self.digit2index[digit] = self.num_tokens
            self.index2digit[self.num_tokens] = digit
            self.digit2count[digit] = 1
            self.num_tokens += 1
            
    def build_vocab(self):        
        for seq in self.digit_seqs:
            for digit in seq:               
                self.add_digit(digit)
            
        # print("Vocabulary created with %d tokens ..." % self.num_tokens)
        # return self.num_tokens
    
    def vocab_size(self):
        return self.num_tokens
    
    def vocabulary(self):
        for idx in self.index2digit:
            print(idx, self.index2digit[idx])
    
    def encode(self, seq):
        return [self.dig2idx(digit) for digit in seq]
    
    def decode(self, seq):
        return "".join([self.idx2dig(idx) for idx in seq])

class MT_Vocab(object):
    def __init__(self, digit_seqs):
        super(MT_Vocab, self).__init__()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        
        # Count SOS, EOS, PAD, UNK
        self.num_words  = 4
        self.digit_seqs = digit_seqs
        
    def word2idx(self, word):
        if word in self.word2index:
            return self.word2index[word]
        
    def idx2word(self, idx):
        if idx in self.index2word:
            return self.index2word[idx]
            
    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
            
        else:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1
            
    def build_vocab(self):        
        for seq in self.digit_seqs:
            for word in seq.split(" "):
                self.add_word(word)

        # print("Vocabulary created with %d tokens ..." % self.num_words)
        # return self.num_tokens
    
    def vocab_size(self):
        return self.num_words
    
    def vocabulary(self):
        for idx in self.index2word:
            print(idx, self.index2word[idx])
    
    def encode(self, seq):
        return [self.word2idx(word) for word in seq.split(" ")]
    
    def decode(self, seq):
        return " ".join([self.idx2word(idx) for idx in seq])
