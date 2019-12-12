"""
Author: Sunyam

This file contains skeletal model architectures: single-input (only Reply Text/Question Text) and double-input (Reply Text and context Comment Text).

Also includes architectures that concatenates Dense Feature Vector for both single-input and double-input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from typing import *
import sys
sys.path.append('../traditional-machine-learning')
import feature_combinations


class SingleInputClassifier(Model):
    """
    Model to classify based on only one input ('Question Text' or 'Reply Text')
    """
    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 class_weights=[4.7, 1.0]):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor(class_weights)) # order is ['yes_unp', 'not_unp']
        # https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch

    # AllenNLP models are required to return a dictionary for every forward pass, and compute the loss function within the forward method during training
    def forward(self, reply_tokens: Dict[str, torch.Tensor],
                ID: List[str], label: torch.Tensor) -> torch.Tensor:

        embeddings = self.word_embeddings(reply_tokens)
        mask = get_text_field_mask(reply_tokens)
        state = self.encoder(embeddings, mask)
#         print("SingleInput embeddings: ", embeddings.size())
#         print("SingleInput state: ", state.size())

        class_logits = self.classifier_feedforward(state)
        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }
        output["loss"] = self.loss(class_logits, torch.max(label, 1)[1]) # because nn.CrossEntropyLoss expects class indices, and not one-hot vectors - https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216
        return output


class DoubleInputClassifier(Model):
    """
    Model to classify based on two input ('Reply Text' & context 'Comment Text')
    """
    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 reply_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 class_weights=[4.7, 1.0]):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.reply_encoder = reply_encoder
        self.context_encoder = context_encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor(class_weights)) # order is ['yes_unp', 'not_unp']

    def forward(self, reply_tokens: Dict[str, torch.Tensor],
                context_tokens: Dict[str, torch.Tensor],
                ID: List[str], label: torch.Tensor) -> torch.Tensor:

        reply_embeddings = self.word_embeddings(reply_tokens)
        reply_mask = get_text_field_mask(reply_tokens)
        reply_state = self.reply_encoder(reply_embeddings, reply_mask)
#         print("Reply embeddings: ", reply_embeddings.size())
#         print("Reply state: ", reply_state.size())

        context_embeddings = self.word_embeddings(context_tokens)
        context_mask = get_text_field_mask(context_tokens)
        context_state = self.context_encoder(context_embeddings, context_mask)
#         print("Context embeddings: ", context_embeddings.size())
#         print("Context state: ", context_state.size())
#         print("Input to Feedforward: ", torch.cat([reply_state, context_state], dim=-1).size())

        class_logits = self.classifier_feedforward(torch.cat([reply_state, context_state], dim=-1))
        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }
        output["loss"] = self.loss(class_logits, torch.max(label, 1)[1])
        return output


def get_dense_vector(df, reply_id, col_name, vectorize_fn):
    """
    For the two Dense classifier classes.
    Get the dense vector for 'col_name' (question/reply/comment) for the row corresponding to 'reply_id' in DataFrame 'df'.
    Calls vectorize_fn to extract feature vector.
    """
    text = df.loc[df['reply_id']==reply_id][col_name].tolist()[0]
    vec = vectorize_fn(text)
    return vec

class DenseSingleClassifier(Model):
    """
    Model to classify based on two inputs (Dense Feature Vector & Reply/Question Text)
    """
    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 col_name: str,
                 class_weights=[4.7, 1.0]):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.col_name = col_name
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor(class_weights)) # order is ['yes_unp', 'not_unp']
        self.df = pd.read_csv('/data/annotations_UQ.csv', lineterminator='\n')

    def forward(self, reply_tokens: Dict[str, torch.Tensor],
                ID: List[str], label: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(reply_tokens)
        mask = get_text_field_mask(reply_tokens)
        state = self.encoder(embeddings, mask)
        X_dense = [get_dense_vector(self.df, i, self.col_name, feature_combinations.embedding_sentiment_ws_lexicon_vector)
                   for i in ID]
        X_dense = torch.from_numpy(np.array(X_dense)).float()
#         print("IDs: ", len(ID), type(ID))
#         print("X_dense: ", X_dense.type(), X_dense.size())
#         print("State: ", state.type(), state.size())
#         print("Input to Feedforward: ", torch.cat([state, X_dense], dim=-1).size())

        class_logits = self.classifier_feedforward(torch.cat([state, X_dense], dim=-1))

        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }
        output["loss"] = self.loss(class_logits, torch.max(label, 1)[1])
        return output


class DenseDoubleClassifier(Model):
    """
    Model to classify based on three inputs (Dense Feature Vector & 'Reply Text' & context 'Comment Text')
    """
    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 reply_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 col_name: str,
                 class_weights=[4.7, 1.0]):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.reply_encoder = reply_encoder
        self.context_encoder = context_encoder
        self.classifier_feedforward = classifier_feedforward
        self.col_name = col_name
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor(class_weights)) # order is ['yes_unp', 'not_unp']
        self.df = pd.read_csv('/data/annotations_UQ.csv', lineterminator='\n')

    def forward(self, reply_tokens: Dict[str, torch.Tensor],
                context_tokens: Dict[str, torch.Tensor],
                ID: str, label: torch.Tensor) -> torch.Tensor:
        # Encode reply:
        reply_embeddings = self.word_embeddings(reply_tokens)
        reply_mask = get_text_field_mask(reply_tokens)
        reply_state = self.reply_encoder(reply_embeddings, reply_mask)
#         print("Reply embeddings: ", reply_embeddings.size())
#         print("Reply state: ", reply_state.size())
        X_dense_reply = [get_dense_vector(self.df, i, self.col_name, feature_combinations.embedding_sentiment_ws_lexicon_vector) for i in ID]
        X_dense_reply = torch.from_numpy(np.array(X_dense_reply)).float()

        # Encode comment text:
        context_embeddings = self.word_embeddings(context_tokens)
        context_mask = get_text_field_mask(context_tokens)
        context_state = self.context_encoder(context_embeddings, context_mask)
#         print("Context embeddings: ", context_embeddings.size())
#         print("Context state: ", context_state.size())
        X_dense_context = [get_dense_vector(self.df, i, 'comment_text', feature_combinations.embedding_sentiment_ws_lexicon_vector) for i in ID]
        X_dense_context = torch.from_numpy(np.array(X_dense_context)).float()

#         print("Input to Feedforward: ", torch.cat([reply_state, X_dense_reply, context_state, X_dense_context], dim=-1).size())

        class_logits = self.classifier_feedforward(torch.cat([reply_state, X_dense_reply, context_state, X_dense_context], dim=-1))
        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }
        output["loss"] = self.loss(class_logits, torch.max(label, 1)[1])
        return output


class InferModel(Model):
    """
    This architecture is inspired from models used in Natural Language Inference.
    Model to classify based on two input ('Reply Text' & context 'Comment Text')
    """
    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 reply_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 class_weights=[4.7, 1.0]):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.reply_encoder = reply_encoder
        self.context_encoder = context_encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor(class_weights)) # order is ['yes_unp', 'not_unp']

    def forward(self, reply_tokens: Dict[str, torch.Tensor],
                context_tokens: Dict[str, torch.Tensor],
                ID: List[str], label: torch.Tensor) -> torch.Tensor:

        reply_embeddings = self.word_embeddings(reply_tokens)
        reply_mask = get_text_field_mask(reply_tokens)
        reply_state = self.reply_encoder(reply_embeddings, reply_mask)

        context_embeddings = self.word_embeddings(context_tokens)
        context_mask = get_text_field_mask(context_tokens)
        context_state = self.context_encoder(context_embeddings, context_mask)

        combined_vector = torch.cat([context_state, reply_state, torch.abs(context_state-reply_state), context_state*reply_state], dim=-1) # concatenation + element-wise-subtraction + multiplication

#         print("Reply state: ", reply_state.size())
#         print("Context state: ", context_state.size())
#         print("Input to Feedforward: ", combined_vector.size())

        class_logits = self.classifier_feedforward(combined_vector)
        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }
        output["loss"] = self.loss(class_logits, torch.max(label, 1)[1])
        return output
