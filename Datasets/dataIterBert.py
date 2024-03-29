from pathlib import Path
from typing import Any, Iterable, Tuple, Callable, List, ClassVar, Dict
import torch
from torch._C import device
from torch.utils.data import Dataset
from transformers import BertTokenizer
from dataIterBase import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase, InferenceDatasetBase


Input = Tuple[str, str]
Batch = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, Batch, torch.IntTensor]


def _get_single_input(query: str, doc: str) -> Input:
    """Return a (query, document) pair for BERT, making sure the strings are not empty
    Args:
        query (str): The query
        doc (str): The document
    Returns:
        Input: Non-empty query and document
    """
    # empty queries or documents might cause problems later on
    if len(query.strip()) == 0:
        query = '(empty)'
    if len(doc.strip()) == 0:
        doc = '(empty)'
    return query, doc


def _collate_bert(inputs: Iterable[Input], tokenizer: BertTokenizer, device: device) -> Batch:
    """Tokenize and collate a number of single BERT inputs, adding special tokens and padding.
    Args:
        inputs (Iterable[Input]): The inputs
        tokenizer (BertTokenizer): Tokenizer
    Returns:
        Batch: Input IDs, attention masks, token type IDs
    """
    queries, docs = zip(*inputs)
    inputs = tokenizer(queries, docs, padding=True, truncation=True)
    return torch.LongTensor(inputs['input_ids']).to(device), \
        torch.LongTensor(inputs['attention_mask']).to(device), \
        torch.LongTensor(inputs['token_type_ids']).to(device)


class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    def __init__(self, data_file: Path, train_file: Path, bert_type: str):
        super().__init__(data_file, train_file)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.
    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """

    def __init__(self, data_file: Path, train_file: Path, bert_type: str, cache_dir: Path):
        super().__init__(data_file, train_file)
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_type, cache_dir=cache_dir)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_bert(pos_inputs, self.tokenizer), _collate_bert(neg_inputs, self.tokenizer)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for BERT validation/testing.
    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """

    def __init__(self, data_file: Path, val_test_file: Path, bert_type: str, cache_dir: Path):
        super().__init__(data_file, val_test_file)
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_type, cache_dir=cache_dir)

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc)

    def collate_fn(self, val_test_inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate a number of validation/testing inputs.
        Args:
            val_test_inputs (Iterable[BertValInput]): The inputs
        Returns:
            ValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
            torch.IntTensor(doc_ids), \
            _collate_bert(inputs, self.tokenizer), \
            torch.IntTensor(labels)


class InferenceDataset(InferenceDatasetBase):

    def __init__(self, documents_file: Path, top_file: Path, bert_type: str, cache_dir: Path, DATA: str, device=torch.device('cpu')):
        super().__init__(documents_file, top_file, DATA=DATA)
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_type, cache_dir=cache_dir)
        self.device = device

    # def __init_q_docs__(self, q_id: str, query: str):
        # pass

    def get_single_input(self, query: str, doc: str) -> Input:
        query, doc = _get_single_input(query, doc)
        inputs = self.tokenizer([query], [doc], truncation=True)
        return torch.LongTensor(inputs['input_ids']).to(self.device), \
            torch.LongTensor(inputs['attention_mask']).to(self.device), \
            torch.LongTensor(inputs['token_type_ids']).to(self.device)


"""
class InferenceDatasetBatch(BatchDataset):

    def __init__(self, documents_file: Path, top_file: Path, bert_type: str, cache_dir: Path, batch_ids: dict, device=torch.device('cpu')):
        super().__init__(documents_file, top_file, batch_ids)
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_type, cache_dir=cache_dir)
        self.device = device

    # def __init_q_docs__(self, q_id: str, query: str):
        # pass

    def get_single_input(self, query: str, doc: str) -> Input:
        query, doc = _get_single_input(query, doc)
        inputs = self.tokenizer([query], [doc], truncation=True)
        return torch.LongTensor(inputs['input_ids']).to(self.device), \
            torch.LongTensor(inputs['attention_mask']).to(self.device), \
            torch.LongTensor(inputs['token_type_ids']).to(self.device)
"""


class CustomDataset(Dataset):
    def __init__(self, query: str, docs: List[str], tokenizer: Any, device=torch.device('cpu')):
        self.query = query
        self.docs = docs
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        query, doc = _get_single_input(self.query, doc)
        return query, doc

    def collate_fn(self, inputs):
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """

        return _collate_bert(inputs, self.tokenizer, self.device)

# necessary changes in __getitem__


class CustomDatasetBatch(Dataset):
    def __init__(self, query: List[str], docs: List[str], tokenizer: Any, device=torch.device('cpu')):
        self.query = query
        self.docs = docs
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        query, doc = _get_single_input(self.query[idx], doc)
        return query, doc  # each query with its top 1 doc

    def collate_fn(self, inputs):
        # Collate a number of pairwise inputs.Args:
        # inputs (Iterable[PairwiseTrainInput]): The inputs
        # Returns:
        # PairwiseTrainBatch: A batch of pairwise inputs

        return _collate_bert(inputs, self.tokenizer, self.device)
