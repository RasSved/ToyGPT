# ToyGPT
Creating an LLM from scratch using pytorch. 
  
dataloader.py  
The very basics of tokenization

bpe.py:  
My own bpe implementation now working. Basically it takes a text and turns it into meaningful byte pairs, these byte pairs is what we later use to represent words and part of unk-words when training our LLM.
