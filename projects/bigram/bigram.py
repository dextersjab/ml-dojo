from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# config
plot_loss = False

# hyperparameters
block_size = 8  # max context length for predictions
batch_size = 4  # num parallel sequences to process
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 200
max_iters = 3000

# Read input text
with open('data/drake_lyrics.txt', 'r', encoding='utf-8') as f:
    source_text = f.read()

# Embed text (as list of sorted characters)
embeddings = list(sorted(set(source_text)))
embeddings_len = len(embeddings)

encoder_mapping = {embedding: i for i, embedding in enumerate(embeddings)}
decoder_mapping = {i: embedding for i, embedding in enumerate(embeddings)}


# Create encoder and decoder
def encode(text: str) -> List[int]:
    """
    Map each character from the input list to the index of its embedding
    in `chars`
    :param text:
    :return:
    """
    return [encoder_mapping[char] for char in text]


def decode(tokens: List[int]) -> str:
    """
    Map each embedded index to its text representation in `chars`
    :param tokens:
    :return:
    """
    return "".join([decoder_mapping[token] for token in tokens])


# Decode source data
data: Tensor = torch.tensor(encode(source_text), dtype=torch.long)

# Split training and validation data 9:1
split_ix = int(.9*len(data))
train_data, val_data = data[:split_ix], data[split_ix:]


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_sequences: Tensor, target_sequences: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Process input tokens and produce network outputs.
        # embeddings (Input): shape (B, T)
        # targets (Optional Input): shape (B, T)
        # [
        #   [token_1, token_2, ..., token_T],  <- sequence 1 (batch 1)
        #   ...
        #   [token_1, token_2, ..., token_T]   <- sequence B (batch B)
        # ]
        The embeddings tensor is passed through the embedding table to produce the logits tensor.
        The logits tensor has the shape (B, T, C), where C is the number of classes (vocab size).

        # logits (Output): shape (B, T, C)
        # [
        #   [
        #       [logit_1_class_1, ..., logit_1_class_C],  <- logit of token 1 (sequence 1)
        #       ...
        #       [logit_T_class_1, ..., logit_T_class_C]   <- logit of token T (sequence 1)
        #   ], <- sequence 1 (batch 1)
        #   ...
        # ]
        If targets are provided (training), computes cross-entropy loss between logits and targets.
        If targets are not provided (evaluation), skips the loss computation.
        :param input_sequences: a (B, T) tensor where B is the batch size and T is the sequence length
        :param target_sequences: a (B, T) tensor of target token ids, or None for inference
        :return: logits (B, T, C) tensor and loss (scalar tensor)
        """
        is_training = target_sequences is not None
        logits = self.token_embedding_table(input_sequences)

        if not is_training:
            loss = None
        else:
            B, T, C = logits.shape
            # Reshape logits and targets for loss computation
            logits = logits.view(B * T, C)
            target_sequences = target_sequences.view(B * T)
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, target_sequences)

        return logits, loss

    def generate(self, input_sequences: Tensor, max_new_tokens: int) -> Tensor:
        output_sequences = input_sequences
        for i in range(max_new_tokens):
            logits, _ = self(output_sequences)
            # All classes (C) in the last sequence (T) of all batches (B) -> (1, C)
            logits = logits[:, -1, :]
            # Transform the last dimension of logits (every class) into relative fractions -> (1, C)
            probs = torch.softmax(logits, dim=-1)
            # Sample one token from the class distribution for each sequence in the batch
            next_tokens = torch.multinomial(probs, num_samples=1)
            # Add the sampled tokens to the existing sequences in the batch
            output_sequences = torch.cat((output_sequences, next_tokens), dim=1)
        return output_sequences


m = BigramLanguageModel(embeddings_len)


def get_batch(is_training: bool = True) -> Tuple[Tensor, Tensor]:
    """
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    Assume batch_size = 3, block_size = 4
    Random starting indices might be [1, 8, 15] (for simplicity)
    The resultant batch (xb) would look something like:
    xb = [
          [1, 2, 3, 4],  <---- sequence 1 (batch 1)
          [8, 9, 10, 11],  <---- sequence 2 (batch 1)
          [15, 16, 17, 18]  <---- sequence 3 (batch 1)
         ]
    Each inner list is a sequence of length 'block_size', and there are 'batch_size' number of such lists.
    :param is_training:
    :return:
    """
    batch_data = train_data if is_training else val_data
    xb_start_ixs: Tensor = torch.randint(len(batch_data) - block_size, (batch_size,))  # generating 'batch_size' random indices
    xb = torch.stack([batch_data[batch_start_index: batch_start_index + block_size] for batch_start_index in xb_start_ixs])
    yb = torch.stack([batch_data[batch_start_index + 1: batch_start_index + block_size + 1] for batch_start_index in xb_start_ixs])
    return xb.to(device), yb.to(device)


zeroes = torch.zeros([1, 1], dtype=torch.long, device=device)
out = m.generate(zeroes, max_new_tokens=100)

optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)


def train(train_steps: int):
    loss = None
    loss_values = []
    for step in range(train_steps):
        # sample a batch of data
        xb, yb = get_batch(is_training=True)

        # compute predictions and loss
        # loss= L(yb, logits) where L is loss function (cross-entropy)
        logits, loss = m(xb, yb)

        # zero out the gradients from the previous step
        # ∇params = 0
        optimiser.zero_grad()

        # backward pass: Compute the gradients of the loss with respect to the model's parameters.
        # ∇params = d(loss)/d(params)
        loss.backward()

        # update the model's params using the computed gradients
        # params' = params - learning_rate * ∇params
        optimiser.step()

        # store the detached loss value for this step
        loss_values.append(loss.detach().item())

        # print loss at intervals
        if step % 500 == 0:
            print(f"loss after {step} training steps: {loss.detach().item()}")

    # print loss for the last batch
    # not necessarily the average or total loss for all batches.
    if loss is not None:
        print(f"loss after {train_steps} training steps: {loss.detach().item()}")
    return loss_values


print(f"{len(embeddings)=}")
print(f"{len(train_data)=}")
print(f"{len(val_data)=}")

train_steps = 10_000
loss_values = train(train_steps=train_steps)

if plot_loss:
    # plot loss values
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label=f"{train_steps} steps")
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss vs training steps')
    plt.legend()
    plt.show()
