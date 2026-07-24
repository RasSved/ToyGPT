# ToyGPT
Creating an LLM from scratch.

# Structure 
## toygpt/tokenizer.py  
dataloader:    
The very basics of tokenization  
  
bpe:  
My own bpe implementation now working. Basically it takes a text and turns it into meaningful byte pairs, these byte pairs is what we later use to represent words and part of unk-words when training our LLM.
  
## toygpt/embeddings.py 
Where we actaully make the vectors from the words, we have straight id to vector and then combined that with the positional vektor to get more context.
  

## toygpt/attention.py
We make an attention score that we later turn into a softmax so we can get better representing vektors. 
