# Transformers from Scratch for Sequence-to-Sequence modelling

This repository contains code to build the transformer model architecture from scratch in PyTorch. The Transformer model, introduced in the paper “Attention is All You Need,” is a deep learning architecture designed for sequence-to-sequence tasks, such as machine translation and text summarization. It is based on self-attention mechanisms and has become the foundation for many state-of-the-art natural language processing models, like GPT and BERT. The transformer architecture is shown in the figure below:

![Transformer Model Architecture](https://github.com/rohitmurali8/Transformers_from_Scratch/blob/master/Transformer.png)

The transformer architecture is essentially an Encoder-Decoder based architecture. One of the most important components present inside the encoder and decoder of the Transformer is the Multi-Head Attention module (MHA). Before we talk about Multi-Head Attention, we start with Self Attention which provides a measure of how words in a sentence are related to each other i.e. it provides a measure of similarity on how every word/token in a sentence is related to a Query. Query basically represents what we are looking for in a sequence, whereas Keys and Values are related to what each word/token in a sentence can offer. The key, query and value vectors for calculating the attention weights are obtained by transforming the input to the model into these vectors. Usually in self attention, there is only one query space that is shared across the input sequence and each token has one query.

Multi-Head attention module is a variation of the attention mechanism which allows for the model to learn multiple aspects of how the words in a sentence are related to each other. This is because in multi-head attention, the queries are projected into a set of different query spaces, allowing parallel attention computations. In MHA, each token has one query per head and multiple heads operate in parallel to enhance the models representational capacity.

The figure below shows how the Multi-Headed Attention architecture looks like:

![Multi-Headed Attention Architecture](https://github.com/rohitmurali8/Transformers_from_Scratch/blob/master/MHA.png)

The implmentation of the transformer model in this repository is done by writing and stacking the following major components:
- Masked/Non-Masked Multi-Head Attention module 
- Feedforward network module
- Positional Encoding
- Encoder module
- Decoder module

For multi-head attention, we use 8 heads to project the queries into 8 different spaces to learn different aspects of the sequence. The encoder and decoder blocks are stacked 6 times to form the complete model. 

For the dataset creation, we will create a dataset which has random source and target data where both have a vocabulary size of 5000 and a maximum sequence length of 100.  
