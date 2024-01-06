from pathlib import Path
from typing import List, Tuple, TextIO
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
import csv
import argparse

root_dir = Path('/home/wang/attackrank/')


def _init_model(dataset: str, rm3: bool) -> SimpleSearcher:
    """ Init pyserini searcher
        args:
            dataset: cluweb09 or robust04.
            rm3: use rm3 query expanding or not
        return:
            pyserini searcher
    """
    if dataset == 'clueweb09':
        index = root_dir / 'Datasets' / 'src' / dataset / 'indexes' / 'clueweb09_indexes'
    elif dataset == 'robust04':
        index = root_dir / 'Datasets' / 'src' / dataset / 'indexes' / 'robust04_indexes'
    else:
        raise ValueError('Only support clueweb09, robust04')

    searcher = SimpleSearcher(str(index))
    if rm3:
        searcher.set_bm25(0.9, 0.4)
        searcher.set_rm3(10, 10, 0.5)
    return searcher


def write_result(target_file: TextIO, result: Tuple[str, List[JSimpleSearcherResult]],
                 hits_num: int, msmarco: bool, tag: str) -> None:
    """write search results to trec_eval like file, copied from pyserini."""
    topic, hits = result
    docids = [hit.docid.strip() for hit in hits]
    scores = [hit.score for hit in hits]

    if msmarco:
        for i, docid in enumerate(docids):
            if i >= hits_num:
                break
            target_file.write(f'{topic}\t{docid}\t{i + 1}\n')
    else:
        for i, (docid, score) in enumerate(zip(docids, scores)):
            if i >= hits_num:
                break
            target_file.write(
                f'{topic} Q0 {docid} {i + 1} {score:.6f} {tag}\n')


def evaluate(dataset: str, rm3: bool, fold_name: str, output: str, hits_num: int) -> None:
    """Evaluate pyserini baseline model performance on fold level
    args:
         dataset: cluweb09 or robust04.
         rm3: use rm3 query expanding or not.
         output: write ranking output for trec_eval.
         hit_num: keep top_{hit_num} results, 100 by default.
    Return:
         None, write output file.
    """
    searcher = _init_model(dataset, rm3)
    query_dir = root_dir / 'Datasets/src/' / dataset / 'queries.tsv'
    queries = {}
    with open(query_dir, encoding='utf-8', newline='') as fp:
        for q_id, query, _, _ in csv.reader(fp, delimiter='\t'):
            queries[q_id] = query

    fold_dir = root_dir / 'Datasets/src/clueweb09/folds' / fold_name / 'test_ids.txt'
    out_dir = Path('/home/lyu/ExpRank/Eval_results/') / fold_name / output

    with open(fold_dir, 'r')as f:
        q_ids = [l.strip() for l in f]

    with open(out_dir, 'w') as f_out:
        for q_id in q_ids:
            query = queries[q_id]
            hits = searcher.search(query, k=hits_num)
            write_result(f_out, (q_id, hits), hits_num, False, output[:-4])
    print(f'Successfully saved eval results to {out_dir}')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('dataset', choices=['clueweb09', 'robust04'], help='Use which dataset.')
    ap.add_argument('fold_name', choices=['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'])
    ap.add_argument('--use_rm3', action='store_true', help='use rm3?')
    ap.add_argument('--output', type=str, help='save to output directory')
    ap.add_argument('--hits_num', type=int, default=100, help='save top_{hits_num} ranking docs')
    args = ap.parse_args()

    evaluate(args.dataset, args.use_rm3, args.fold_name, args.output, args.hits_num)


if __name__ == '__main__':
    main()

