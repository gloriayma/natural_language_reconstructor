input inversion - it's not hard to train a model to do input inversion on itself - how hard is it to train another model to do input inversion on it? how much finetuning until a model can't really do input inversion on itself? 

what models can i train from scratch (no pretraining) to try to do input inversion? 
- test: same architecture, different architecture? 

how easy is it to input inversion a small model to a bigger model? the other way around? with or without pretraining? 

those are the motivating questions. 

experiment 1: to establish a baseline and a training procedure, we take a pretrained model and we train (finetune) an identical instantiation of that model to do input inversion. 
concretely, what i want is: 
- model 1 

deliverable: list of models (not too many because training is expensive - max 4, can you get away with 3 or 2?)