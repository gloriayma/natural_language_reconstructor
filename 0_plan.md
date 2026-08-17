in open source language models (the best one), is the embedding layer the inverse of the unembedding layer? that is to say - if i embed a token and then unembed it, does it give me    
  the token back? otherwise, is it operating in the same "space" - would it give the next token prediction dictated by bigram frequency? is there anything tying the embedding layer to   
  the unembedding layer         

i would like you to run experiments over a couple of models - of different sizes, of different model families, whatever is free and trainable on this cluster - i would like the best models to be the best that my compute can handle running inference on (with activation hooks); and a good sample of models that my compute can handle running finetuning on. models that have tied weights, models that don't, etc

as a first experiment, i want to have a sample of words
- this sample of words should be quite diverse and ideally random but stratified. i want common and proper nouns, verbs, preopositions, other parts of speech
    - they should not all be one token, but some should be one token and some should be more than one
    - also give me some weird tokens, including punctuation and EOM or padding tokens. also give me the weirdest ones that are still used in english, but hold out anything that would just be confusing for me (put them in the list, but not in the final list of tokens / words). include EOM / padding / header tokens in the set (don't exclude them), not just the weird set

first, i want you to pass in these words to the model, and grab their embeddings after the embedding layer. pass these words in fresh each time (no context). if it is a multi-token word, grab each token involved. 

next, i want you to run these tokens through the model's unembedding layer. 

for each model, get the embedding(s) (deterministic), then feed it through the unembedding layer for the final output logits (deterministic), and tell me about the top k 

and that is the final deliverable.

intermediate deliverables:
- list of models chosen from the broader open-source model space, with reasons
- list of words / tokens chosen 


notes
- please make a new conda env named interp and install whatever you need in there
- 