"""
Author: Sunyam

This file is for reading the unpalatable question dataset as AllenNLP 'Instances'.
"""

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField #, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

from typing import *
from overrides import overrides
import numpy as np
import pandas as pd


class UnpalatableDatasetReader(DatasetReader):
    """
    Read the annotated dataset of unpalatable questions:
    - 'main_input' should be either 'reply_text' or 'question'
    - 'additional_context' should be True if preceding comment text is to be considered
    """
    def __init__(self, main_input: str, additional_context: bool,
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer]=None,
                 label_cols: List[str]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_cols = label_cols
        self.main_input = main_input # this captures whether to run it on Reply Text or Question Text
        self.additional_context = additional_context # True if (preceding) Comment Text is to be used as well
        
    @overrides
    def text_to_instance(self, reply_tokens: List[Token], 
                         context_tokens: List[Token]=None,
                         ID: str=None,
                         labels: np.ndarray=None) -> Instance:
        
        reply = TextField(reply_tokens, self.token_indexers)
        fields = {"reply_tokens": reply}
        
        if self.additional_context and context_tokens is not None: # should consider context Comment Text
            context_comment = TextField(context_tokens, self.token_indexers)
            fields["context_tokens"] = context_comment
        
        id_field = MetadataField(ID)
        fields["ID"] = id_field
        
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path, lineterminator='\n')

        if self.additional_context: # consider Comment Text as well
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    reply_tokens=[Token(x) for x in self.tokenizer(row[self.main_input])],
                    context_tokens=[Token(x) for x in self.tokenizer(row['comment_text'])],
                    ID=row["reply_id"], # unique ID for every row
                    labels=row[self.label_cols].values
                )
        
        else: # only consider 'main_input'
            for i, row in df.iterrows():
                yield self.text_to_instance(
                    reply_tokens=[Token(x) for x in self.tokenizer(row[self.main_input])],
                    context_tokens=None,
                    ID=row["reply_id"], # unique ID for every row
                    labels=row[self.label_cols].values
                )