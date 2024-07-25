#!/usr/bin/env python
# coding: utf-8

# # Text Generator
# Implementing a text generation model from scratch using a transformer (decoder only).\
# Steps:
# 1. Tokenization
# 2. Input embedding
# 3. Positional encoding
# 4. Masking
# 5. Self-attention
# 6. Decoder stack
# 7. Predicting token probabilities

# ## Creating Training Data

# In[5]:


# !nvidia-smi


# In[6]:


import torch
import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[7]:


print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# In[8]:


class creating_data():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def save(self, path):
        self.df.to_csv(path)


# In[9]:


# dataset = creating_data('medium_articles.csv')
# dataset.save('training_data.csv')


# ## Tokenization

# In[10]:


class Tokenizer():
    def __init__(self):
        '''
        Initializes the tokenizer with a basic character vocabulary
        '''
        # initializing two dictionaries for forward and reverse tokenization
        self.dictionary = {}
        self.reverse_dictionary = {}

        # adding digits, alphabets, and basic punctuation to the vocabulary
        self.__add_to_dict('<pad>')
        self.__add_to_dict('<unk>')

        for i in range(10):
            self.__add_to_dict(str(i))

        for i in range(26):
            self.__add_to_dict(chr(ord('a') + i))
            self.__add_to_dict(chr(ord('A') + i))

        for char in ['.', ' ', ',', '!', '?', '\n']:
            self.__add_to_dict(char)

    def __add_to_dict(self, character):
        '''
        Adds a character to the dictionary and reverse dictionary
        Args: 
            character: the character to add.
        '''
        if character not in self.dictionary:
            index = self.size()
            self.dictionary[character] = index
            self.reverse_dictionary[index] = character

    def tokenize(self, text):
        """
        Converts a text string into a list of token indices.

        Args:
          text: The input text string.

        Returns:
          A list of token indices corresponding to the characters in the text.
        """
            return [self.character_to_token(character) for character in text]

    def character_to_token(self, character):
        """
        Converts a character to its corresponding token index.

        Args:
          character: The input character.

        Returns:
          The token index for the character, or the index of the '<unk>' token if not found.
        """
        return self.dictionary.get(character, self.dictionary['<unk>'])

    def token_to_character(self, token):
        """
        Converts a token index to its corresponding character.

        Args:
          token: The input token index.

        Returns:
          The character corresponding to the token index, or '<unk>' if not found.
        """
            return self.reverse_dictionary.get(token, '<unk>')

    def size(self):
        """
        Returns the size of the vocabulary (number of unique characters).

        Returns:
          The size of the vocabulary.
        """
        return len(self.dictionary)


# ## Input Embeddings

# In[11]:


class TokenEmbedding(torch.nn.Module):
    """
      Converts input token indices into embedding vectors.

      Attributes:
        embedding_layer: A PyTorch Embedding layer for converting tokens to embeddings.
      """

    def __init__(self, model_dim, num_tokens):
        """
        Initializes the TokenEmbedding module.

        Args:
          model_dim: The dimensionality of the output embeddings.
          num_tokens: The size of the input vocabulary (number of unique tokens).
        """
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings = num_tokens,
            embedding_dim = model_dim
        )

    def forward(self, x):
        """
        Converts input token indices into embedding vectors.

        Args:
          x: A tensor of token indices (shape: [batch_size, sequence_length]).

        Returns:
          A tensor of embedding vectors (shape: [batch_size, sequence_length, model_dim]).
        """
        return self.embedding_layer(x)


# ## Positional Encoding

# In[12]:


class PositionalEncoding(torch.nn.Module):
    """
      Adds positional encoding to input embeddings.

      Attributes:
        model_dim: The dimensionality of the input and output embeddings.
        max_sequence_length: The maximum length of input sequences.
        positional_encoding: A pre-computed tensor of positional encodings.
      """
    def __init__(self, model_dim, max_sequence_length):
        """
        Initializes the PositionalEncoding module.

        Args:
          model_dim: The dimensionality of the input and output embeddings.
          max_sequence_length: The maximum length of input sequences.
        """
        super().__init__()
        self.model_dim = model_dim
        self.max_sequence_length = max_sequence_length
        positional_encoding = np.zeros((max_sequence_length, model_dim))

        # calculating positional encoding for each position and dimension
        for pos in range(max_sequence_length):
            for i in range(0, self.model_dim, 2):
                # sine for even indices
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / model_dim)))

                # cosine for odd indices
                if i + 1 < self.model_dim:
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / model_dim)))

        self.positional_encoding = torch.from_numpy(positional_encoding).float().to(get_device())

    def forward(self, x):
        """
        Adds positional encoding to input embeddings.

        Args:
          x: Input embedding tensor (shape: [batch_size, sequence_length, model_dim]).

        Returns:
          Output tensor with pos
        """
        return x + self.positional_encoding[: x.size(1), :]


# ## Masking and Attention

# In[13]:


class MaskedSelfAttention(torch.nn.Module):
    """
      Performs masked self-attention on input embeddings.

      Attributes:
        embedding_dimension: Dimensionality of input embeddings.
        head_dimension: Dimensionality of attention heads.
        query_layer: Linear layer for projecting inputs into query space.
        key_layer: Linear layer for projecting inputs into key space.
        value_layer: Linear layer for projecting inputs into value space.
        softmax: Softmax activation function.
      """
    def __init__(self, embedding_dimension, head_dimension):
        """
        Initializes the MaskedSelfAttention module.

        Args:
          embedding_dimension: Dimensionality of input embeddings.
          head_dimension: Dimensionality of attention heads.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension

        self.query_layer = torch.nn.Linear(self.embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(self.embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(self.embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x, mask):
        """
        Performs masked self-attention on input embeddings.

        Args:
          x: Input embeddings (shape: [batch_size, sequence_length, embedding_dimension]).
          mask: Attention mask (shape: [batch_size, sequence_length, head_dimension]).

        Returns:
          Output tensor after applying self-attention (shape: [batch_size, sequence_length, head_dimension]).
        """
        
        # projects inputs into query, key, and value vectors
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # calculating attention weights and scaling
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dimension)

        # applying masking to prevent attending to future tokens
        if mask is not None:
            mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
            attention_weights = attention_weights.masked_fill(mask == 0, -1e8)
        
        # softmax normalization
        attention_scores = self.softmax(attention_weights)
        
        # Weighted sum of values
        return torch.bmm(attention_scores, value)


# In[14]:


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
      Performs multi-head self-attention on input embeddings.

      Attributes:
        embedding_dimension: Dimensionality of input embeddings.
        head_dimension: Dimensionality of each attention head (embedding_dimension // num_heads).
        num_heads: Number of attention heads.
        self_attentions: A list of MaskedSelfAttention modules, one for each head.
        output_layer: Linear layer for projecting the concatenated attention outputs back to the original embedding dimension.
      """
    def __init__(self, embedding_dimension, num_heads):
        """
        Initializes the MaskedMultiHeadedSelfAttention module.

        Args:
          embedding_dimension: Dimensionality of input embeddings.
          num_heads: Number of attention heads.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // num_heads
        self.num_heads = num_heads
        
        # Creating a list of MaskedSelfAttention modules, one for each head
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(self.num_heads)]
        )

        # Linear layer to project the concatenated attention outputs back to original dimension
        self.output_layer = torch.nn.Linear(self.num_heads * self.head_dimension, self.embedding_dimension)

    def forward(self, x, mask):
        """
        Performs multi-head self-attention on input embeddings.

        Args:
          x: Input embeddings (shape: [batch_size, sequence_length, embedding_dimension]).
          mask: Attention mask (shape: [batch_size, sequence_length, head_dimension]).

        Returns:
          Output tensor after applying multi-head self-attention (shape: [batch_size, sequence_length, embedding_dimension]).
        """
        # applying self attention independently on each head
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

         # Concatenating the outputs from each head along the feature dimension
        concatenated_outputs = torch.cat(self_attention_outputs, dim = 2)
        
        # projecting the outputs back to the original embedding dimension
        return self.output_layer(concatenated_outputs)


# ## Decoder

# In[ ]:


class FeedForward(torch.nn.Module):
    """
      Feed-forward neural network with two linear layers and a ReLU activation.

      Attributes:
        linear_1: First linear layer.
        linear_2: Second linear layer.
      """
    def __init__(self, embedding_dim, feed_forward_dim):
        """
        Initializes the FeedForward module.

        Args:
          embedding_dim: Dimensionality of the input and output embeddings.
          feed_forward_dim: Dimensionality of the hidden layer.
        """
        super().__init__()
        self.linear_1 = torch.nn.Linear(embedding_dim, feed_forward_dim)
        self.linear_2 = torch.nn.Linear(feed_forward_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the feed-forward network.

        Args:
          x: Input tensor (shape: [batch_size, sequence_length, embedding_dim]).

        Returns:
          Output tensor (shape: [batch_size, sequence_length, embedding_dim]).
        """
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)

        return x


# In[15]:


class DecoderLayer(torch.nn.Module):
    """
      A single layer in the decoder network.

      Attributes:
        multi_attention: A MaskedMultiHeadedSelfAttention module for self-attention within the decoder.
        feed_forward: A FeedForward module for non-linear processing.
        dropout: A dropout layer for regularization.
        layer_norm_1: Layer normalization applied before the multi-head self-attention.
        layer_norm_2: Layer normalization applied before the feed-forward sublayer.
      """
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout_rate):
        """
        Initializes the DecoderLayer module.

        Args:
          embedding_dim: Dimensionality of input and output embeddings.
          num_heads: Number of attention heads in the multi-head self-attention layer.
          feed_forward_dim: Dimensionality of the hidden layer in the feed-forward sublayer.
          dropout_rate: Dropout rate for regularization.
        """
        super().__init__()

        self.multi_attention = MaskedMultiHeadedSelfAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.layer_norm_1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x, mask):
        """
        Performs a single decoder layer operation.

        Args:
          x: Input tensor (shape: [batch_size, sequence_length, embedding_dim]).
          mask: Attention mask to prevent attending to future positions (shape: [batch_size, sequence_length, 1]).

        Returns:
          Output tensor after processing through the decoder layer (shape: [batch_size, sequence_length, embedding_dim]).
        """
        # applying layer normalization before self-attention
        x_norm = self.layer_norm_1(x)
        attention_output = self.multi_attention(x_norm, mask)
        # residual connection with the input
        residual_output = x + attention_output

        # feedforward block with layer normalization and ReLU
        residual_output_norm = self.layer_norm_2(residual_output)
        feed_forward_output = self.feed_forward(residual_output_norm)

        # applying dropout during training for regularization
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)
    
        # adding the outputs from the residual connection and feed-forward sublayers
        return residual_output + feed_forward_output


# In[16]:


class DecoderStack(torch.nn.Module):
    """
      A stack of decoder layers.

      Attributes:
        max_sequence_length: Maximum sequence length for positional encoding.
        decoder_layers: A list of DecoderLayer modules.
      """
    def __init__(self, embedding_dim, num_layers, num_heads, feed_forward_dim, dropout_rate, max_sequence_length):
        """
        Initializes the DecoderStack module.

        Args:
          embedding_dim: Dimensionality of input and output embeddings.
          num_layers: Number of decoder layers.
          num_heads: Number of attention heads in each decoder layer.
          feed_forward_dim: Dimensionality of the feed-forward sublayer in each decoder layer.
          dropout_rate: Dropout rate for regularization.
          max_sequence_length: Maximum sequence length for positional encoding.
        """
        super().__init__()

        self.max_sequence_length = max_sequence_length
        
        # Creating a list of DecoderLayer modules
        self.decoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, feed_forward_dim, dropout_rate) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        """
        Processes the input through multiple decoder layers.

        Args:
          x: Input tensor (shape: [batch_size, sequence_length, embedding_dim]).
          mask: Attention mask to prevent attending to future positions (shape: [batch_size, sequence_length, 1]).

        Returns:
          Output tensor after processing through all decoder layers (shape: [batch_size, sequence_length, embedding_dim]).
        """
        outputs = x
        for layer in self.decoder_layers:
            outputs = layer(outputs, mask)

        return outputs


# ## Building the Model

# In[ ]:


class GeneratorHead(torch.nn.Module):
    """
      Maps the final decoder state to output logits.

      Attributes:
        linear: Linear layer to project the decoder output to the vocabulary size.
      """
    def __init__(self, embedding_dim, num_tokens):
        """
        Initializes the GeneratorHead module.

        Args:
          embedding_dim: Dimensionality of the input embeddings.
          num_tokens: Size of the vocabulary.
        """
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, num_tokens)

    def forward(self, x):
        """
        Projects the decoder output to logits.

        Args:
          x: Decoder output (shape: [batch_size, sequence_length, embedding_dim]).

        Returns:
          Logits for each token in the vocabulary (shape: [batch_size, sequence_length, num_tokens]).
        """
        return self.linear(x)


# In[35]:


class TextGenerator(torch.nn.Module):
    class TextGenerator(torch.nn.Module):
      """
      Transformer-based model for text generation.

      This class implements a text generation model using a Transformer architecture with a decoder stack.

      Attributes:
        num_tokens: Size of the vocabulary (number of unique words).
        max_sequence_length: Maximum length of input sequences.
        embedding_dim: Dimensionality of word embeddings.
        num_layers: Number of decoder layers in the Transformer architecture.
        num_heads: Number of attention heads in each decoder layer.
        feed_forward_dim: Dimensionality of the hidden layer in the feed-forward sublayer within each decoder layer (defaults to 4 times embedding_dim).
        dropout_rate: Dropout rate for regularization during training.
      """
    def __init__(self, num_tokens, max_sequence_length = 100, embedding_dim = 512, num_layers = 6, num_heads = 4, feed_forward_dim = None, dropout_rate = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        if feed_forward_dim is None:
            self.feed_forward_dim = embedding_dim * 4
        else:
            self.feed_forward_dim = feed_forward_dim

        self.dropout_rate = dropout_rate
        
        # Creating submodules for embedding, positional encoding, layer normalization
        self.token_embedding = TokenEmbedding(embedding_dim, num_tokens)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_sequence_length)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

        # Creating the decoder stack and generator head
        self.decoder = DecoderStack(embedding_dim, num_layers, num_heads, self.feed_forward_dim, dropout_rate, max_sequence_length)
        self.generator_head = GeneratorHead(embedding_dim, num_tokens)

    def forward(self, x, mask):
        """
        Performs the text generation process.

        Args:
          x: Input sequence of token indices (shape: [batch_size, sequence_length]).
          mask: Attention mask to prevent attending to future positions (shape: [batch_size, sequence_length, 1]).

        Returns:
          Logits for each token in the vocabulary at each position in the sequence (shape: [batch_size, sequence_length, num_tokens]).
        """

        token_embedding = self.token_embedding(x)
        positional_encoding = self.positional_encoding(token_embedding)
        positional_encoding_norm = self.layer_norm(positional_encoding)
        
        # Passing through decoder layers for processing
        decoder_outputs = self.decoder(positional_encoding_norm, mask)
        # Projecting the decoder output to vocabulary logits
        generator_outputs = self.generator_head(decoder_outputs)

        return generator_outputs

    def save_checkpoint(self, filepath):
        """
        Saves the model checkpoint.

        Args:
          filepath: Path to save the checkpoint.
        """
        print(f'Saving checkpoint {filepath}')
        torch.save({
            'number_of_tokens': self.num_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dim,
            'number_of_layers': self.num_layers,
            'number_of_heads': self.num_heads,
            'feed_forward_dimension': self.feed_forward_dim,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, filepath)

    @staticmethod
    def load_checkpoint(filepath):
        """
        Loads a saved model checkpoint.

        Args:
          filepath: Path to the checkpoint file.

        Returns:
          An instance of the TextGenerator model.
        """
        checkpoint = torch.load(filepath).to(get_device())
        model = TextGenerator(
            num_tokens = checkpoint['number_of_tokens'],
            max_sequence_length = checkpoint['max_sequence_length'],
            embedding_dim = checkpoint['embedding_dimension'],
            num_layers = checkpoint['number_of_layers'],
            num_heads = checkpoint['number_of_heads'],
            feed_forward_dim = checkpoint['feed_forward_dimension'],
            dropout_rate = checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(get_device())


# ## Autoregressive Wrapper

# In[20]:


class AutoregressiveWrapper(torch.nn.Module):
    """
      Wrapper class to enable autoregressive generation with the text generator model.

      This class provides methods for training and generating text using the underlying text generator model.

      Attributes:
        model: The text generator model instance.
        max_sequence_length: Maximum length of sequences handled by the model.
      """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.max_sequence_length = self.model.max_sequence_length

    def forward(self, x, mask):
        """
        Prepares input and target sequences for training.

        Args:
          x: Input sequence of token indices (shape: [batch_size, sequence_length]).
          mask: Attention mask to prevent attending to future positions (shape: [batch_size, sequence_length, 1]).

        Returns:
          Tuple of input and target sequences.
        """
        inputs, targets = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inputs, mask)
        return output, targets

    def next_token_probabilities(self, x, mask, temperature = 1.0):
        """
        Calculates probabilities for the next token given the current input.

        Args:
          x: Input sequence of token indices (shape: [batch_size, sequence_length]).
          mask: Attention mask to prevent attending to future positions (shape: [batch_size, sequence_length, 1]).
          temperature: Temperature for controlling randomness in sampling.

        Returns:
          Probability distribution over the vocabulary for the next token.
        """
        logits = self.model(x, mask)[:, -1]

        logits /= temperature

        probabilities = torch.softmax(logits, dim = -1)

        return probabilities

    def save_checkpoint(self, filepath):
        """
        Saves the model checkpoint.

        Args:
          filepath: Path to save the checkpoint.
        """
        self.model.save_checkpoint(filepath)

    @staticmethod
    def load_checkpoint(filepath):
        """
        Loads a saved model checkpoint.

        Args:
          filepath: Path to the checkpoint file.

        Returns:
          An instance of the AutoregressiveWrapper with the loaded model.
        """
        model = TextGenerator.load_checkpoint(filepath)
        return AutoregressiveWrapper(model).to(get_device())


# In[21]:


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Training the Model

# In[22]:


class Trainer:
    """
      Class for training the text generation model.

      This class handles training the `TextGenerator` model on a provided dataset.

      Attributes:
        model: The text generation model instance.
        tokenizer: A tokenizer object for converting text to token indices and vice versa.
        optimizer: The optimizer used for training (defaults to Adam with learning rate 0.001).
        loss_function: The loss function used for calculating training loss (defaults to cross-entropy loss).
      """
    def __init__(self, model, tokenizer: Tokenizer, optimizer = None):
        self.model = model

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        else:
            self.optimizer = optimizer

        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, data, epochs, batch_size):
        """
        Trains the model on the provided dataset.

        Args:
          data: List of training sequences (represented as lists of token indices).
          epochs: Number of training epochs.
          batch_size: Size of training batches.

        Returns:
          List of average losses per epoch during training.
        """
        loss_epoch = [] # tracking loss for each epoch

        for epoch in range(epochs):
            losses = []
            # shuffling the data for each epoch
            random.shuffle(data)
            
            # creating batches from the shuffled data
            batches = []
            for i in range(0, len(data), batch_size):
                sequence = torch.tensor(data[i: i + batch_size], dtype = torch.long)
                # Creating attention mask: 0 for padding tokens, 1 otherwise
                mask_tensor = torch.ones_like(sequence)
                mask_tensor[sequence == self.tokenizer.character_to_token('<pad>')] = 0

                batches.append((sequence, mask_tensor))

            # Training progress bar
            epoch_progress = tqdm(batches, desc = f"Epoch {epoch + 1}/{epochs}", unit = "batch")
            for batch in epoch_progress:
                # setting the model to training mode
                self.model.train()
                
                # creating the input and mask tensors, and handling the padding
                input_tensor = torch.zeros((batch_size, self.model.model.max_sequence_length + 1), dtype = torch.long)
                mask_tensor = torch.zeros((batch_size, self.model.model.max_sequence_length + 1), dtype = torch.long)

                for i, inp in enumerate(batch[0]):
                    input_tensor[i] = inp

                for i, mask in enumerate(batch[1]):
                    mask_tensor[i] = mask
                
                # forward pass, and calculating loss
                model_output, target = self.model.forward(x = input_tensor.to(get_device()), mask = mask_tensor.to(get_device()))

                loss = self.loss_function(model_output.transpose(1, 2), target)
                
                # backpropogation and optimization
                loss.backward()    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())

            # calculaitng and printing loss for each epoch
            epoch_loss = np.average(losses)
            loss_epoch.append(epoch_loss)
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        return loss_epoch


# ## Generator

# In[23]:


class Generator:
    """
      Class for generating text using the trained text generation model.

      This class provides methods for generating text sequences based on a prompt and temperature parameter.

      Attributes:
        model: The text generation model instance.
        tokenizer: A tokenizer object for converting text to token indices and vice versa.
      """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def pad_left(self, sequence, final_length, padding_token):
        """
        Pads a sequence with the specified padding token to a given final length.

        Args:
          sequence: List of tokens.
          final_length: Desired length of the padded sequence.
          padding_token: Token used for padding.

        Returns:
          The padded sequence.
        """
        return [padding_token] * (final_length - len(sequence)) + sequence

    def generate(self, max_tokens, prompt = None, temperature = 1.0, eos_token = None, padding_token = 0):
        """
        Generates a text sequence using the model.

        Args:
          max_tokens: Maximum number of tokens to generate.
          prompt: Optional starting prompt for the generation (list of tokens or None).
          temperature: Controls randomness in sampling (higher temperature leads to more diverse outputs).
          eos_token: Optional end-of-sequence token (if set, generation stops when encountered).
          padding_token: Token used for padding during generation.

        Returns:
          The generated text sequence as a string.
        """
        self.model.eval() # setting model to evaluation mode

        # handling the starting prompt
        if prompt is None:
            start_tokens = [self.tokenizer.character_to_token(padding_token)]
        else:
            start_tokens = self.tokenizer.tokenize(prompt)
        
        # preparing the input sequence with padding
        input_tokens = self.pad_left(start_tokens, self.model.max_sequence_length, padding_token)

        input_tensor = torch.tensor(
            input_tokens, dtype = torch.long
        ).to(get_device())
        
        # adding batch dimensions
        dims = len(input_tensor.shape)
        if dims == 1:
            input_tensor = input_tensor[None, :]

        out = input_tensor
        generated = input_tokens[:] # keeping track of generated tokens

        for _ in range(max_tokens):
            # getting the most recent sequence for prediction
            x = out[:, -self.model.max_sequence_length:]
            
            # creating attention masks
            mask = torch.ones_like(x)
            mask[x == padding_token] = 0
            
            # getting the probabilities for the next token and sampling bask on this
            next_token_prob = self.model.next_token_probabilities(x = x, temperature = temperature, mask = mask)
            #print(next_token_prob)
            next_token = torch.multinomial(next_token_prob, num_samples = 1).item()
            #print(next_token)
            generated.append(next_token)
        
            # stopping if the end of sequence token is generated
            if eos_token is not None and next_token == eos_token:
                break
                
            # updating the sequence with the sampled token
            new_token_tensor = torch.tensor([[next_token]], dtype = torch.long).to(get_device())
            out = torch.cat((out, new_token_tensor), dim = 1)

        #generated_tokens = input_tensor[0].tolist()
        # converting the generated tokens back to text
        return ''.join([self.tokenizer.token_to_character(token) for token in generated])


# ## Running

# In[24]:


def create_training_sequences(max_sequence_length, tokenized_data):
    """
      Creates training sequences by sliding a window over the tokenized data.

      Args:
        max_sequence_length: The desired length of each training sequence.
        tokenized_data: A list of tokenized text data.

      Returns:
        A list of training sequences, where each sequence is a list of tokens.
      """
    sequences = []

    for i in range(0, len(tokenized_data) - max_sequence_length - 1):
        sequences.append(tokenized_data[i: i + max_sequence_length + 1])

    return sequences


# In[25]:


def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    """
      Tokenizes and pads training data for input to the model.

      Args:
        max_sequence_length: The desired maximum length of each sequence.
        tokenizer: A tokenizer object for converting text to tokens.
        training_data: The raw text data.

      Returns:
        A list of tokenized and padded sequences.
      """
    tokenized_data = tokenizer.tokenize(training_data)

    for _ in range(max_sequence_length):
        tokenized_data.insert(0, tokenizer.character_to_token('<pad>'))

    return tokenized_data


# In[26]:


class Run(torch.nn.Module):
    """
      Class for training and running a text generation model.

      This class handles creating the model, training it on provided data, and generating text based on prompts.
      """

    def __init__(self, embedding_dim = 256, max_sequence_length = 50):
        """
        Initializes the class with hyperparameters and creates placeholders for model and tokenizer.

        Args:
          embedding_dim: Dimensionality of word embeddings (default: 256).
          max_sequence_length: Maximum length of sequences handled by the model (default: 50).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.model = None
        self.tokenizer = None

    def train_model(self):
        """
        Trains the text generation model on a dataset.

        1. Creates a tokenizer for converting text to tokens.
        2. Determines the vocabulary size based on the tokenizer.
        3. Creates a text generation model with specified hyperparameters.
        4. Loads training data from a CSV file (limited to 10 samples for demonstration).
        5. Preprocesses the training data by tokenization and padding.
        6. Creates training sequences by sliding a window over the preprocessed data.
        7. Creates a trainer object with optimizer and trains the model for 100 epochs with a batch size of 32.
        8. Plots the training loss per epoch in log scale.
        9. Saves the trained model to a checkpoint file.
        """
        self.tokenizer = Tokenizer()
        num_tokens = self.tokenizer.size()

        self.model = AutoregressiveWrapper(TextGenerator(
            embedding_dim = self.embedding_dim,
            num_tokens = num_tokens,
            num_heads = 4,
            num_layers = 3,
            dropout_rate = 0.1,
            max_sequence_length = self.max_sequence_length
        )).to(get_device())

        training_data = pd.read_csv('training_data.csv')['text'].tolist()[:10]
        # joining samples with sentence separators
        training_data = '. '.join(training_data)

        tokenized_and_padded_training_data = tokenize_and_pad_training_data(self.max_sequence_length, self.tokenizer, training_data)
        sequences = create_training_sequences(self.max_sequence_length, tokenized_and_padded_training_data)

        # training
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        trainer = Trainer(self.model, self.tokenizer, optimizer)
        loss_per_epoch = trainer.train(sequences, epochs = 100, batch_size = 32)

        # Plot the loss per epoch in log scale
        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        self.model.save_checkpoint('./trained_model')

    def run(self, prompt):
        """
        Generates text using the trained model based on a prompt.

        Args:
          prompt: Starting text sequence (string) to provide context for generation.

        Returns:
          Generated text as a string.
        """

        # generate text
        max_tokens = 1000
        generator = Generator(self.model, self.tokenizer)
        generated_text = generator.generate(
            max_tokens = max_tokens, prompt = prompt, padding_token = self.tokenizer.character_to_token('<pad>')
        )

        print(generated_text.replace('<pad>', ''))


# In[36]:


runner = Run()
runner.train_model()


# In[37]:


runner.run(prompt = "Photo by")


# In[38]:


runner.run(prompt = "Merry Christ")


# In[39]:


runner.run(prompt = "Potato")


# In[40]:


runner.run(prompt = "Tell me about a time")


# In[42]:


runner.run(prompt = "We just wanted everyone to know how much we appreciate everyone and how thankful we are for all our readers and writers here")

