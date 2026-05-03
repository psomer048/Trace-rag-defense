import numpy as np
import re
import torch
import json
import zlib
import os
import joblib
import csv
from typing import List, Dict, Set, Tuple, Any, Callable, Optional
from transformers import BertModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast 
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from src.poison_feature_utils import (
    LEGACY_FEATURE_NAMES,
    build_group_feature_map,
    feature_vector_from_map,
    unpack_model_artifact,
)
from src.scorer_paths import ACTIVE_POISON_SCORER_REL_PATH, resolve_scorer_path

class DataCollector:
    def __init__(self, filepath="poison_training_data.csv", feature_names: Optional[List[str]] = None):
        self.filepath = filepath
        self.feature_names = list(feature_names or LEGACY_FEATURE_NAMES)
        self.headers = self.feature_names + ["label"]
        # ,
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def save_sample(self, features: List[float], label: int):
        # features headers
        row = features + [label]
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class PoisoningScorer:
    def __init__(self, model_path=ACTIVE_POISON_SCORER_REL_PATH):
        self.model = None
        self.requested_model_path = model_path
        self.model_path = resolve_scorer_path(model_path)
        self.feature_names = list(LEGACY_FEATURE_NAMES)
        self.feature_set = "legacy"
        self.threshold = None
        self.metrics = {}
        self.load_model()

        print(f"[PoisoningScorer] Requested model path: {self.requested_model_path}")
        print(f"[PoisoningScorer] Attempting to load model from: {self.model_path}")
        
        if os.path.exists(self.model_path):
            try:
                artifact = unpack_model_artifact(joblib.load(self.model_path))
                self.model = artifact["model"]
                self.feature_names = artifact["feature_names"]
                self.feature_set = artifact["feature_set"]
                self.threshold = artifact["threshold"]
                self.metrics = artifact["metrics"]
                print(f"[PoisoningScorer] SUCCESS: Loaded ML model.")
                print(
                    f"[PoisoningScorer] Feature schema: {self.feature_set} "
                    f"({len(self.feature_names)} dims)"
                )
                # Translated comment (English only).
                if hasattr(self.model, "n_features_in_"):
                    print(f"[PoisoningScorer] Model expects {self.model.n_features_in_} features.")
            except Exception as e:
                print(f"[PoisoningScorer] ERROR loading model: {e}")
        else:
            print(f"[PoisoningScorer] WARNING: Model file not found at {self.model_path}. Using Rule-Based mode.")


    def predict(self, features: List[float]) -> float:
        if self.model is None: return -1.0
        try:
            if hasattr(self.model, "predict_proba"):
                return float(self.model.predict_proba([features])[0][1])
            else:
                return float(self.model.predict([features])[0])
        except Exception as e:
            # Translated comment (English only).
            print(f"[PoisoningScorer] Prediction Error: {e}")
            print(f"  Input Features: {features}")
            return -1.0

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                artifact = unpack_model_artifact(joblib.load(self.model_path))
                self.model = artifact["model"]
                self.feature_names = artifact["feature_names"]
                self.feature_set = artifact["feature_set"]
                self.threshold = artifact["threshold"]
                self.metrics = artifact["metrics"]
                print(f"[PoisoningScorer] Loaded ML model from {self.model_path}")
            except Exception as e:
                print(f"[PoisoningScorer] Error loading model: {e}")
        else:
            print(f"[PoisoningScorer] No model found at {self.model_path}. Using Rule-Based mode.")


class PoisoningDetector:
    def __init__(
        self,
        embedding_model=None,
        tokenizer=None,
        get_emb_fn=None,
        debug: bool = False,
        logger=None,
        model_path: str = "poison_rf.pkl",
        collect_data_mode: bool = False,
        data_filepath: str = "poison_training_data.csv",
    ):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.get_emb_fn = get_emb_fn
        
        # debug
        self.debug = debug
        self.logger = logger if logger is not None else print

        # PPL
        self.ppl_model = None
        self.ppl_tokenizer = None
        self.sbert_model = None
        
        # --- ML ---
        self.scorer = PoisoningScorer(model_path=model_path)
        collector_features = (
            list(self.scorer.feature_names)
            if getattr(self.scorer, "feature_names", None)
            else list(LEGACY_FEATURE_NAMES)
        )
        self.collector = (
            DataCollector(filepath=data_filepath, feature_names=collector_features)
            if collect_data_mode
            else None
        )
        self.collect_data_mode = collect_data_mode

        # ---  Stage 1 Embedding ---
        self.sbert_model = None
        try:
            from sentence_transformers import SentenceTransformer
            if debug: self.logger("[Init] Loading Sentence-BERT for Stage 1 clustering...")
            # (20MB ),, Stage 1
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Warning: Failed to load Sentence-BERT: {e}")

        # ---  Cross-Encoder ---
        # self.ce_model = None 

    def _dbg(self, msg: str):
        if self.debug:
            self.logger(msg)

    # --- ---
    def _lazy_load_ppl_model(self, device):
        if self.ppl_model is None:
            try:
                if self.debug:
                    self.logger("[Init] Loading GPT-2 for Perplexity check...")
                self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                self.ppl_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.ppl_model.to(device)
                self.ppl_model.eval()
            except Exception as e:
                print(f"Failed to load GPT-2: {e}")
                self.ppl_model = None

    def _simple_tokenize(self, text: str) -> List[str]:
        return re.findall(r'\S+', text)

    def _calculate_compression_ratio(self, text: str) -> float:
        if not text: return 0.0
        try:
            bytes_data = text.encode('utf-8')
            original_size = len(bytes_data)
            if original_size == 0: return 0.0
            compressed_data = zlib.compress(bytes_data)
            compressed_size = len(compressed_data)
            return compressed_size / original_size
        except Exception:
            return 0.0

    def _is_suspicious_token(self, token: str) -> bool:
        if len(token) > 25: return True
        non_alnum = sum(1 for c in token if not c.isalnum())
        if len(token) > 10 and (non_alnum / len(token)) > 0.4: return True
        if 6 < len(token) <= 25:
            for i in range(1, len(token) // 3 + 1):
                sub = token[:i]
                repeats = token.count(sub)
                if repeats >= 3 and len(sub) * repeats >= len(token) * 0.8:
                    return True
        return False

        # --- Stage 0: Gradient Selection (Full) ---
    def stage0_token_alignment_check(self, query: str, documents: List[Dict], N: int = 10) -> None:
        if not documents: return
        if not isinstance(self.embedding_model, BertModel):
            if self.debug: self.logger("Stage0 check skipped: Not a BertModel.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(device)
        self.embedding_model.eval() 

        # 1. Query Embedding
        try:
            q_inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            with torch.no_grad():
                q_out = BertModel.forward(self.embedding_model, **q_inputs).last_hidden_state
                q_mask_expanded = q_inputs['attention_mask'].unsqueeze(-1).expand(q_out.size()).float()
                q_sum = torch.sum(q_out * q_mask_expanded, 1)
                q_counts = torch.clamp(q_mask_expanded.sum(1), min=1e-9)
                q_emb = q_sum / q_counts
                q_emb = F.normalize(q_emb, p=2, dim=1)
        except Exception as e:
            print(f"Stage0 Query Emb Error: {e}")
            return

        # 2. Word Embeddings
        try:
            word_embeddings = self.embedding_model.embeddings.word_embeddings
        except AttributeError:
            return

        all_max_grads = []
        
        for doc in documents:
            text = doc.get("text", doc.get("context", ""))
            
            if not isinstance(text, str) or not text.strip():
                self._update_doc_stage0_empty(doc)
                all_max_grads.append(0.0)
                continue

            try:
                self.embedding_model.zero_grad()
                d_inputs = self.tokenizer(text, padding=False, truncation=True, return_tensors="pt")
                d_inputs = {k: v.to(device) for k, v in d_inputs.items()}
                d_input_ids = d_inputs['input_ids']

                # Detach embedding layer to compute gradients on input tokens
                d_embeds = word_embeddings(d_input_ids).detach()
                d_embeds.requires_grad = True

                # Forward pass using inputs_embeds
                d_out_obj = BertModel.forward(
                    self.embedding_model,
                    inputs_embeds=d_embeds,
                    attention_mask=d_inputs['attention_mask'],
                    return_dict=True
                )
                d_out = d_out_obj.last_hidden_state
                d_mask_expanded = d_inputs['attention_mask'].unsqueeze(-1).expand(d_out.size()).float()
                d_sum = torch.sum(d_out * d_mask_expanded, 1)
                d_counts = torch.clamp(d_mask_expanded.sum(1), min=1e-9)
                d_emb = d_sum / d_counts
                d_emb = F.normalize(d_emb, p=2, dim=1)

                # Compute similarity score and backward pass
                score = torch.sum(q_emb * d_emb)
                score.backward()

                # Get gradient norms
                grads = d_embeds.grad[0].norm(dim=1)
                grad_mean_threshold = grads.mean()
                mask = grads > grad_mean_threshold
                selected_indices = torch.nonzero(mask).squeeze()
                
                # Handle edge cases for dimensions
                if selected_indices.ndim == 0: selected_indices = selected_indices.unsqueeze(0)
                
                if selected_indices.numel() > 0:
                    selected_grads = grads[selected_indices]
                    current_k = min(N, selected_grads.numel())
                    top_k_values, top_k_indices_local = torch.topk(selected_grads, current_k)
                    final_indices = selected_indices[top_k_indices_local]
                    final_grads = top_k_values.detach().cpu().numpy()
                    final_indices_cpu = final_indices.detach().cpu().numpy()
                else:
                    final_grads, final_indices_cpu = [], []

                # Map back to tokens
                all_tokens = self.tokenizer.convert_ids_to_tokens(d_input_ids[0].cpu().tolist())
                token_grad_list = []
                valid_max_grads = []

                for idx, g_val in zip(final_indices_cpu, final_grads):
                    if idx < len(all_tokens):
                        tok_str = all_tokens[idx]
                        if tok_str in ['[CLS]', '[SEP]', '[PAD]']: continue
                        token_grad_list.append({"token": tok_str, "grad": float(g_val)})
                        valid_max_grads.append(float(g_val))

                max_grad = valid_max_grads[0] if valid_max_grads else 0.0
                mean_grad = np.mean(valid_max_grads) if valid_max_grads else 0.0

                doc.update({
                    "stage0_max_grad": float(max_grad),
                    "stage0_mean_grad": float(mean_grad),
                    "stage0_top_trigger_tokens": token_grad_list
                })
                all_max_grads.append(float(max_grad))

            except Exception as e:
                # Fallback for errors
                self._update_doc_stage0_empty(doc)
                all_max_grads.append(0.0)

        # 3. Calculate Z-Scores
        self._calculate_z_scores(documents, all_max_grads, key_prefix="stage0_grad")

    def _update_doc_stage0_empty(self, doc):
        doc.update({
            "stage0_max_grad": 0.0, "stage0_mean_grad": 0.0,
            "stage0_top_trigger_tokens": [], "stage0_grad_z": 0.0
        })

    # --- Stage 1: Semantic Micro-clustering (Robust Version) ---
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        # SBERT, Stage 1
        if self.sbert_model is not None:
            return self.sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
        if not self.embedding_model or not self.tokenizer or not self.get_emb_fn:
            raise ValueError("No valid embedding model available for Stage 1")
            
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            # model nn.Module .to(),
            if hasattr(self.embedding_model, 'to'):
                self.embedding_model.to('cuda')
        with torch.no_grad():
            embs = self.get_emb_fn(self.embedding_model, inputs)
            if isinstance(embs, torch.Tensor):
                embs = embs.cpu().numpy()
        return embs

    def _get_stage1_focus_text(self, text: str, max_sentences: int = 2) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        normalized = re.sub(r"\s+", " ", text)
        sentence_like = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+", normalized)
            if segment.strip()
        ]
        if sentence_like:
            return " ".join(sentence_like[:max_sentences])

        # Fallback for punctuation-poor passages: keep a short leading window.
        tokens = normalized.split()
        if len(tokens) <= 80:
            return normalized
        return " ".join(tokens[:80])

    def stage1_semantic_check(self, query: str, viewpoints: List[Dict], retrieve_fn: Callable) -> None:
        """
        Stage 1: Check for multi-document attacks via semantic micro-clustering.
        Only the viewpoint's final-answer documents are scored here.
        Similarity is computed on each document's leading answer-bearing prefix
        (roughly the first two sentences) instead of the full document.
        """
        del query, retrieve_fn

        for group in viewpoints:
            analysis_docs = list(group.get('docs', []))
            group['support_docs'] = []

            if not analysis_docs:
                self._set_stage1_empty(group)
                continue

            texts = []
            for doc in analysis_docs:
                raw_text = doc.get('text', doc.get('context', ''))
                focus_text = self._get_stage1_focus_text(raw_text)
                doc['stage1_focus_text'] = focus_text
                texts.append(focus_text)
            try:
                embs = self._get_embeddings(texts)
                if embs.shape[0] <= 1:
                    self._set_stage1_empty(group)
                    continue
                
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-9)
                sim_matrix = np.dot(embs, embs.T)
                np.fill_diagonal(sim_matrix, -1.0)
                
                nn_sims = np.max(sim_matrix, axis=1)
                
                # NN-Sim
                for i, doc in enumerate(analysis_docs):
                    doc['stage1_nn_sim'] = float(nn_sims[i])
                
                # Slightly tighten the echo-cluster trigger to reduce clean biomedical
                # abstracts being treated as micro-clones while preserving strong poison clusters.
                echo_nn_med_ratio = float(np.mean(nn_sims >= 0.90))
                
                centroid = np.mean(embs, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                dists_to_centroid = 1.0 - np.dot(embs, centroid)
                radius_mean = float(np.mean(dists_to_centroid))
                
                group['stage1_features'] = {
                    'echo_nn_med_ratio': echo_nn_med_ratio,
                    'radius_mean': radius_mean,
                    'max_nn_sim': float(np.max(nn_sims)),
                }
                
                suspicious_count = sum(1 for d in analysis_docs if any(self._is_suspicious_token(t) for t in self._simple_tokenize(d.get('text', d.get('context', '')))))
                group['stage1_suspicious_ratio'] = suspicious_count / len(analysis_docs)
                
            except Exception as e:
                # ,
                if self.debug: self.logger(f"Stage 1 Error VP{idx}: {e}")
                self._set_stage1_empty(group)

    def _set_stage1_empty(self, group):
        group['stage1_features'] = {'echo_nn_med_ratio': 0.0, 'radius_mean': 1.0, 'max_nn_sim': 0.0}
        group['stage1_suspicious_ratio'] = 0.0
        for doc in group.get('docs', []): doc['stage1_nn_sim'] = 0.0

    # --- Stage 2: Structural Features ---
    def stage2_structural_features(self, query: str, documents: List[Dict]) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lazy_load_ppl_model(device)

        q_emb = None
        try:
            q_emb = self._get_embeddings([query])[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        except Exception: pass

        overlaps, cos_sims, ppl_scores, comp_ratios = [], [], [], []

        for doc in documents:
            text = doc.get('text', doc.get('context', ''))
            
            overlap = self._ngram_overlap(query, text, n=3)
            overlaps.append(overlap)
            doc['stage2_ngram_overlap_q'] = overlap
            
            cos_sim = 0.0
            if q_emb is not None:
                try:
                    d_emb = self._get_embeddings([text])[0]
                    d_emb = d_emb / (np.linalg.norm(d_emb) + 1e-9)
                    cos_sim = float(np.dot(q_emb, d_emb))
                except Exception: pass
            cos_sims.append(cos_sim)
            doc['stage2_cos_sim_q'] = cos_sim

            ppl = 0.0
            if self.ppl_model is not None and text.strip():
                try:
                    encodings = self.ppl_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
                    encodings = {k: v.to(device) for k, v in encodings.items()}
                    with torch.no_grad():
                        outputs = self.ppl_model(**encodings, labels=encodings['input_ids'])
                        ppl = torch.exp(outputs.loss).item()
                        if np.isnan(ppl) or np.isinf(ppl): ppl = 10000.0
                except Exception: ppl = 0.0
            ppl_scores.append(ppl)
            doc['stage2_ppl'] = ppl
            
            comp_ratio = self._calculate_compression_ratio(text)
            doc['stage2_compression_ratio'] = comp_ratio
            comp_ratios.append(comp_ratio)

        self._calculate_z_scores(documents, overlaps, "stage2_ngram_overlap_q")
        self._calculate_z_scores(documents, cos_sims, "stage2_cos_sim_q")
        self._calculate_z_scores(documents, ppl_scores, "stage2_ppl")

    def _ngram_overlap(self, text1: str, text2: str, n: int = 3) -> float:
        tokens1 = self._simple_tokenize(text1.lower())
        tokens2 = self._simple_tokenize(text2.lower())
        if len(tokens1) < n or len(tokens2) < n: return 0.0
        ngrams1 = set(tuple(tokens1[i:i+n]) for i in range(len(tokens1)-n+1))
        ngrams2 = set(tuple(tokens2[i:i+n]) for i in range(len(tokens2)-n+1))
        if not ngrams1: return 0.0
        return len(ngrams1.intersection(ngrams2)) / len(ngrams1)

    def _calculate_z_scores(self, documents, values, key_prefix):
        if not values or len(values) < 2:
            for doc in documents:
                key = f"{key_prefix}_z"
                doc[key] = 0.0
            return
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        for i, doc in enumerate(documents):
            val = values[i]
            z = (val - mean_val) / (std_val + 1e-9) if std_val > 1e-9 else 0.0
            doc[f"{key_prefix}_z"] = z

    # --- Stage 3: Cross-Encoder (Disabled) ---
    def stage3_anomaly_detection(self, query: str, documents: List[Dict]) -> None:
        """Stage 3 Disabled"""
        for doc in documents:
            doc['stage3_ce_score'] = 0.0
            doc['stage3_ce_trigger_drop'] = 0.0

        # --- Stage 4: Aggregation (Hybrid: ML + Rules + Collection) ---
    def stage4_aggregation(self, viewpoints: List[Dict]) -> None:
        if self.debug:
            self.logger("\n" + "="*60)
            self.logger(" [Stage 4] Detailed Arbitration & Scoring (Hybrid Mode)")
            self.logger("="*60)

        for idx, group in enumerate(viewpoints):
            docs = group.get('docs', [])
            claim = group.get('claim', 'Unknown')
            if not docs:
                group['suspicion_score'] = 0.0
                continue

            # 1. (Base Statistics)
            feature_map = build_group_feature_map(group)
            group_size = float(feature_map['group_size'])
            avg_grad_z = float(feature_map['avg_grad_z'])
            avg_ppl = float(feature_map['avg_ppl'])
            high_ppl_ratio = float(feature_map['high_ppl_ratio'])
            avg_z_overlap = float(feature_map['avg_overlap_z'])
            high_overlap_ratio = float(feature_map['high_overlap_ratio'])
            echo_nn_med_ratio = float(feature_map['echo_med'])
            radius_mean = float(feature_map['radius'])
            max_nn_sim = float(feature_map['max_sim'])
            mean_nn_sim = float(feature_map.get('mean_nn_sim', 0.0))
            stage1_suspicious_ratio = float(feature_map['susp_ratio'])

            active_feature_names = list(getattr(self.scorer, "feature_names", LEGACY_FEATURE_NAMES))
            features = feature_vector_from_map(feature_map, active_feature_names)

            # 2. (Auto-Labeling for Training)
            if self.collect_data_mode:
                # Label logic: If >50% docs are poisoned, label=1
                poison_cnt = sum(1 for d in docs if d.get('is_poison', False))
                label = 1 if poison_cnt > 1 else 0
                self.collector.save_sample(features, label)
                if self.debug:
                    label_str = "POISON" if label else "CLEAN"
                    # self.logger(f"    [Data Collected] Label: {label_str}")

            # 3. (Scoring Logic)
            score = 0.0
            reasons = []
            
            # === Option A: Machine Learning Prediction ===
            ml_prob = self.scorer.predict(features)
            
            if ml_prob >= 0:
                # ,
                score = ml_prob
                reasons.append(f"ML_Model({score:.2f})")
            else:
                # === Option B: Rule-Based Fallback () ===
                hard_poison = False

                # Rule 1: Gradient Anomaly
                grad_threshold = 3.0 if group_size == 1 else 2.5
                if group_size >= 1 and avg_grad_z > grad_threshold:
                    hard_poison = True
                    reasons.append(f"strong_grad_anomaly(avg_z={avg_grad_z:.2f})")
                
                # Rule 2: High Perplexity
                if group_size >= 1 and high_ppl_ratio > 0.7:
                    hard_poison = True
                    reasons.append(f"strong_high_ppl(ratio={high_ppl_ratio:.2f})")

                # Rule 3: Strong Echo Chamber (Tuned)
                # :>3 MaxSim>0.85
                if group_size > 3 and max_nn_sim > 0.85:
                    hard_poison = True
                    reasons.append(f"strong_echo_chamber(max_sim={max_nn_sim:.2f})")
                
                # Rule 3.5: Fallback Echo (Combination)
                # Rule 3,
                elif group_size >= 3 and echo_nn_med_ratio >= 0.4 and radius_mean < 0.40:
                    hard_poison = True
                    reasons.append(f"dense_cluster(r={radius_mean:.2f})")

                # Rule 4: Strong Query Stuffing
                if group_size >= 3 and high_overlap_ratio >= 0.6 and avg_z_overlap >= 1.0:
                    hard_poison = True
                    reasons.append(f"strong_query_stuffing")

                if hard_poison:
                    score = 1.0
                else:
                    # Soft Rules
                    if group_size > 1 and avg_grad_z > 1.2:
                        score += 0.3; reasons.append("mod_grad")
                    if avg_ppl > 80.0: 
                        score += 0.2; reasons.append("mod_ppl")
                    if echo_nn_med_ratio > 0.3 and radius_mean < 0.5:
                        score += 0.3; reasons.append("mod_echo") # Weight increased
                    if stage1_suspicious_ratio > 0.3:
                        score += 0.2; reasons.append("susp_tokens")
                    score = min(1.0, score)

            if group_size >= 3 and high_ppl_ratio >= 0.75 and (radius_mean <= 0.35 or mean_nn_sim >= 0.60):
                bonus = 0.4
                score = min(1.0, score + bonus)
                reasons.append(
                    f"adaptive_high_ppl_cluster_bonus(+{bonus:.2f},ppl_ratio={high_ppl_ratio:.2f},"
                    f"radius={radius_mean:.2f},mean_nn={mean_nn_sim:.2f})"
                )

            # Store Results
            group['stage4_group_features'] = {
                'avg_grad_z': avg_grad_z, 'avg_ppl': avg_ppl, 
                'score_reasons': reasons,
                'feature_schema': getattr(self.scorer, 'feature_set', 'legacy'),
                'active_feature_names': active_feature_names,
                'active_feature_vector': features,
                'all_features': feature_map,
                'echo_nn_med_ratio': echo_nn_med_ratio,
                'radius_mean': radius_mean,
                'max_nn_sim': max_nn_sim,
                'mean_nn_sim': mean_nn_sim,
            }
            group['suspicion_score'] = score
            
            # --- Detailed Logging ---
            if self.debug:
                status = "POISONED" if score > 0.8 else ("SUSPICIOUS" if score > 0.5 else "CLEAN")
                self.logger(f"\n>>> Viewpoint {idx+1} [Docs: {int(group_size)}] Claim: '{claim[:50]}...'")
                self.logger(f"    [Group Stats] GradZ: {avg_grad_z:.2f} | PPL: {avg_ppl:.1f} | Echo-Med: {echo_nn_med_ratio:.2f} | Radius: {radius_mean:.2f} | MaxSim: {max_nn_sim:.2f}")
                self.logger(f"    [Result] {status} (Score: {score:.2f}) | Reasons: {reasons}")
                
                self.logger(f"    [Document Details]")
                for doc in docs:
                    d_id = doc.get('id', 'unk')
                    d_ppl = doc.get('stage2_ppl', 0.0)
                    d_gz = doc.get('stage0_grad_z', 0.0)
                    d_comp = doc.get('stage2_compression_ratio', 0.0)
                    d_nnsim = doc.get('stage1_nn_sim', 0.0)
                    self.logger(f"      - {d_id}: PPL={d_ppl:.1f}, GradZ={d_gz:.2f}, NNSim={d_nnsim:.2f}, Comp={d_comp:.2f}")

    def arbitrate_viewpoints(self, query: str, viewpoints: List[Dict], retrieve_fn: Callable) -> Dict:
        # ... (Stage 0, 2, 1, 4 ) ...
        all_docs = []
        for group in viewpoints: all_docs.extend(group.get('docs', []))
        
        # token
        # self.stage0_token_alignment_check(query, all_docs) 
        self.stage2_structural_features(query, all_docs)   
        self.stage1_semantic_check(query, viewpoints, retrieve_fn)
        self.stage4_aggregation(viewpoints)
        
        # ===  ===
        
        results = []
        scores = []
        
        # 1.
        for group in viewpoints:
            scores.append(group.get('suspicion_score', 0.0))

        # 2.
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        score_gap = max_score - min_score
        
        # :,(>0.9),
        # :VP1=0.68, VP2=0.92 -> Gap=0.24 -> VP1
        rescue_threshold = 0.85  # ,,
        gap_requirement = 0.2    # Translated comment (English only).
        
        target_status = [] # Translated comment (English only).
        
        for s in scores:
            if s > 0.8: # (Hard Poison)
                target_status.append("poisoned")
            elif s > 0.6:
                # (0.6  0.8)
                # ,,
                if s == min_score and score_gap > gap_requirement and s < rescue_threshold:
                    target_status.append("clean_rescued") # Translated comment (English only).
                else:
                    target_status.append("suspicious")
            else:
                target_status.append("clean")

        # 3.
        clean_viewpoints = 0
        all_suspicious = True
        
        for idx, group in enumerate(viewpoints):
            status = target_status[idx]
            s_score = scores[idx]
            
            # rescued clean
            real_status = "clean" if status == "clean_rescued" else status
            
            if real_status == "clean":
                all_suspicious = False
                clean_viewpoints += 1
            
            results.append({
                'claim': group.get('claim'), 
                'suspicion_score': s_score, 
                'status': real_status, # Translated comment (English only).
                'details': group.get('stage4_group_features')
            })
            
        # 4.
        if not viewpoints: overall = "no_viewpoints"
        elif all_suspicious: overall = "all_poisoned"
        elif clean_viewpoints == len(viewpoints): overall = "clean_contradiction"
        elif clean_viewpoints > 0: overall = "mixed_poisoned"
        else: overall = "all_poisoned"
            
        return {'viewpoint_analysis': results, 'overall_conclusion': overall}
