# Includes the class definitions for ASR vocbulary

# Default word tokens
BLK_token = 0  # Blank label
PAD_token = 1  # Used for padding short utterances

class ASR_Vocab(object):
    def __init__(self, digit_seqs):
        super(ASR_Vocab, self).__init__()
        self.digit2index = {}
        self.digit2count = {}
        self.index2digit = {BLK_token: "BLK", PAD_token: "PAD"}
        
        # Count SOS, EOS, PAD, EMP
        self.num_tokens  = 2
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
            for digit in seq:      # Ignore EOS token
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
    