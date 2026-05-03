import logging
import os
import json
import time
from pathlib import Path

import hashlib

import torch
import transformers

from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


from src.contriever_src.contriever import Contriever
from src.contriever_src.beir_utils import DenseEncoderModel
from src.utils import load_beir_datasets, load_models
from src.dataset_profiles import get_dataset_dir, resolve_split

import argparse
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model_code', type=str, default="contriever")
parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--data_path', type=str, default=None, help='Optional dataset directory override')

parser.add_argument('--result_output', default="results/beir_results/debug.json", type=str)

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument("--use_embedding_cache", action="store_true", help="Use a persistent local corpus embedding cache for retrieval.")
parser.add_argument("--embedding_cache_dir", default="results/embedding_cache", type=str)

args = parser.parse_args()

from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_cache_key(dataset_name, data_path, model_code, max_length):
    source = os.path.abspath(data_path or dataset_name)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{dataset_name}-{model_code}-max{max_length}-{digest}"


def encode_texts(texts, encoder_model, tokenizer, get_emb_fn, batch_size, max_length, device):
    embeddings = []
    nbatch = (len(texts) - 1) // batch_size + 1
    with torch.no_grad():
        for batch_idx in range(nbatch):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            encoded = tokenizer.batch_encode_plus(
                texts[start_idx:end_idx],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            emb = get_emb_fn(encoder_model, encoded)
            embeddings.append(emb.detach().cpu())
    return torch.cat(embeddings, dim=0)


def retrieve_with_embedding_cache(args, corpus, queries, score_function, device):
    cache_root = Path(args.embedding_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_cache_key(args.dataset, args.data_path or "", args.model_code, args.max_length)
    cache_path = cache_root / f"{cache_key}.pt"

    model, c_model, tokenizer, get_emb = load_models(args.model_code)
    model.eval()
    c_model.eval()
    model.to(device)
    c_model.to(device)

    if cache_path.exists():
        logging.info("Loading cached corpus embeddings from %s", cache_path)
        cache_payload = torch.load(cache_path, map_location="cpu")
        doc_ids = cache_payload["doc_ids"]
        doc_embeddings = cache_payload["embeddings"].float()
    else:
        logging.info("Building corpus embedding cache at %s", cache_path)
        doc_ids = list(corpus.keys())
        corpus_texts = [
            f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}".strip()
            if corpus[doc_id].get("title")
            else corpus[doc_id].get("text", "")
            for doc_id in doc_ids
        ]
        doc_embeddings = encode_texts(
            corpus_texts,
            c_model,
            tokenizer,
            get_emb,
            args.per_gpu_batch_size,
            args.max_length,
            device,
        ).float()
        torch.save(
            {
                "doc_ids": doc_ids,
                "embeddings": doc_embeddings.half(),
                "dataset": args.dataset,
                "data_path": os.path.abspath(args.data_path or ""),
                "model_code": args.model_code,
                "max_length": args.max_length,
            },
            cache_path,
        )

    if score_function == "cos_sim":
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=1)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = encode_texts(
        query_texts,
        model,
        tokenizer,
        get_emb,
        args.per_gpu_batch_size,
        args.max_length,
        device,
    ).float()
    if score_function == "cos_sim":
        query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

    doc_embeddings_t = doc_embeddings.t()
    top_k = min(args.top_k, len(doc_ids))
    results = {}
    retrieval_loop_start = time.time()
    for idx, query_id in enumerate(query_ids):
        scores = torch.matmul(query_embeddings[idx:idx + 1], doc_embeddings_t).squeeze(0)
        values, indices = torch.topk(scores, k=top_k)
        results[query_id] = {
            doc_ids[int(doc_idx)]: float(score)
            for score, doc_idx in zip(values.tolist(), indices.tolist())
        }
        processed = idx + 1
        if processed % 500 == 0 or processed == len(query_ids):
            elapsed = time.time() - retrieval_loop_start
            avg = elapsed / processed
            eta = avg * (len(query_ids) - processed)
            logging.info(
                "[Embedding Cache Retrieval] processed %s/%s queries | elapsed=%s | eta=%s",
                processed,
                len(query_ids),
                format_duration(elapsed),
                format_duration(eta),
            )
    return results

def compress(results):
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

logging.info(args)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### Download and load dataset
dataset = args.dataset
args.split = resolve_split(args.dataset, args.split)
data_path = get_dataset_dir(dataset, data_path=args.data_path)
logging.info(data_path)

corpus, queries, qrels = load_beir_datasets(args.dataset, args.split, data_path=args.data_path)
logging.info(
    "Loaded dataset stats | queries=%s | corpus=%s | qrels=%s",
    len(queries),
    len(corpus),
    len(qrels),
)

# grp: If you want to use other datasets, you could prepare your dataset as the format of beir, then load it here.

retrieval_start = time.time()
logging.info("Starting retrieval at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(retrieval_start)))
if args.use_embedding_cache:
    logging.info("Using persistent embedding cache retrieval path.")
    results = retrieve_with_embedding_cache(args, corpus, queries, args.score_function, device)
else:
    logging.info("Loading model...")
    if 'contriever' in args.model_code:
        encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
        model = DRES(
            DenseEncoderModel(
                encoder,
                doc_encoder=encoder,
                tokenizer=tokenizer,
                max_length=args.max_length,
            ),
            batch_size=args.per_gpu_batch_size,
        )
    elif 'dpr' in args.model_code:
        try:
            from beir.retrieval.models import DPR
        except ImportError as exc:
            raise ImportError(
                "The installed BEIR version does not expose DPR. "
                "Please switch model_code or install a BEIR version with DPR support."
            ) from exc
        model = DRES(DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])), batch_size=args.per_gpu_batch_size, corpus_chunk_size=5000)
    elif 'ance' in args.model_code:
        model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
    else:
        raise NotImplementedError

    logging.info(f"model: {model.model}")

    retriever = EvaluateRetrieval(model, score_function=args.score_function, k_values=[args.top_k]) # "cos_sim"  or "dot" for dot-product
    results = retriever.retrieve(corpus, queries)
retrieval_elapsed = time.time() - retrieval_start
avg_per_query = retrieval_elapsed / max(len(queries), 1)
logging.info(
    "Retrieval finished in %s (%.2fs/query)",
    format_duration(retrieval_elapsed),
    avg_per_query,
)
                                            
logging.info("Printing results to %s"%(args.result_output))
sub_results = compress(results)

os.makedirs(os.path.dirname(args.result_output), exist_ok=True)
with open(args.result_output, 'w') as f:
    json.dump(sub_results, f)
