#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Gayathri Devi Thotappa
# #### Student ID: s4111690
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * numpy
# * sklearn
# * tqdm
# * gensim 
# 
# ## Introduction
# This notebook completes Task 2 (features) and Task 3 (classification) for the clothing-reviews dataset (~19.6k items). From the cleaned Review Text (Task 1), we build:
# 
# Task 2:
# 
# * Bag-of-Words (Count vectors) using the Task-1 vocabulary → count_vectors.txt.
# 
# * Embeddings: one model (FastText) as unweighted and TF-IDF-weighted document vectors.
# 
# Task 3:
# 
# Q1: Compare BoW vs. unweighted vs. TF-IDF-weighted embeddings with the same classifier.
# 
# Q2: Test information gain using BoW on Title only, Text only, and Title + Text.
# 
# Protocol: 5-fold stratified CV with Logistic Regression, reporting Accuracy, Precision (macro), Recall (macro), F1 (macro) 

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import os, re, ast
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer 
from gensim.models import KeyedVectors


# ## Task 2. Generating Feature Representations for Clothing Items Reviews

# Convert the cleaned Review Text from Task 1 into three document representations (Title ignored as specified):
# 
# Bag-of-Words (Count vectors): built strictly from vocab.txt; saved in sparse format as count_vectors.txt 
# 
# Embeddings — Unweighted: average of word vectors from the chosen model (FastText); saved as fasttext_unweighted.csv.
# 
# Embeddings — TF-IDF Weighted: weighted average using TF-IDF to emphasise informative words; saved as fasttext_tfidf.csv.
# These representations balance interpretability (BoW) and semantic signal (embeddings), and will be used unchanged in Task 3.

# In[2]:


# Task 2 uses only the cleaned Review Text from Task 1 (Title is ignored here as per spec).
#  lock CountVectorizer's vocabulary to it for consistent indices from vocab.txt 
PROCESSED_CSV_PATH = "processed.csv"             # from Task-1
VOCAB_TXT_PATH     = "vocab.txt"                 # from Task-1  

COUNT_VECTORS_TXT  = "count_vectors.txt"         # required output format

# Download, unzip, and point to the .vec file:
#Link to download fasttext vec pretrained model: https://fasttext.cc/docs/en/english-vectors.html 
FASTTEXT_VEC_PATH  = "wiki-news-300d-1M-subword.vec"

FT_UNW_NPY         = "fasttext_unweighted.npy"
FT_UNW_CSV         = "fasttext_unweighted.csv"
FT_TFIDF_NPY       = "fasttext_tfidf.npy"
FT_TFIDF_CSV       = "fasttext_tfidf.csv"

# processed.csv contains tokenized text so set TOKEN_COL:
TOKEN_COL  = 'tokens'            
REVIEW_COL = "Review Text"   # used if TOKEN_COL is None

# regex from Task-1 spec
TOKEN_REGEX = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"


# In[3]:


#tokenization: regex + lowercasing.
def tokenize_with_regex(text: str):
    if not isinstance(text, str): return []
    return [t.lower() for t in re.findall(TOKEN_REGEX, text)]


# In[4]:


#If tokens are stored as list-like strings, parse safely. Else fall back to space split.
def parse_tokens_cell(x):
    if isinstance(x, list):
        return [str(t).lower() for t in x]
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return [str(t).lower() for t in ast.literal_eval(s)]
            except Exception:
                return [tok.lower() for tok in s.split()]
        return [tok.lower() for tok in s.split()]
    return [] 


# In[5]:


# Load processed data
assert os.path.exists(PROCESSED_CSV_PATH), "processed.csv not found."
df = pd.read_csv(PROCESSED_CSV_PATH)

# Decide how to read tokens  
if TOKEN_COL and TOKEN_COL in df.columns:
    docs_tokens = df[TOKEN_COL].apply(parse_tokens_cell).tolist()
else:
    assert REVIEW_COL in df.columns, f"'{REVIEW_COL}' not found in processed.csv"
    docs_tokens = df[REVIEW_COL].fillna("").apply(tokenize_with_regex).tolist()

n_docs = len(docs_tokens)
print(f"Loaded {n_docs} documents from processed.csv")


# In[6]:


# Load Task-1 vocabulary
# Expected format per line: "word:idx"
assert os.path.exists(VOCAB_TXT_PATH), "vocab.txt not found."
v2i = {}
with open(VOCAB_TXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s: continue
        w, idx = s.split(":")
        v2i[w] = int(idx)

V = len(v2i)
i2v = {i:w for w,i in v2i.items()}
assert set(v2i.values()) == set(range(V)), "Vocab indices must be contiguous 0..V-1"
print(f"Loaded vocab size = {V}") 


# In[7]:


# Write sparse Count vectors
def doc_sparse_counts(tokens):
    c = Counter(t for t in tokens if t in v2i)         # restrict to Task-1 vocab
    return sorted(((v2i[w], f) for w, f in c.items()), key=lambda x: x[0])  # index-ascending

 


# In[8]:


# Build TF-IDF 
# pre-tokenised → pass space-joined text, provide vocabulary mapping
docs_space = [" ".join([t for t in toks if t in v2i]) for toks in docs_tokens]

tfidf = TfidfVectorizer(
    vocabulary=v2i,        # FIXED mapping (word -> column index)
    lowercase=False,
    tokenizer=str.split,   # tokens are space-separated already
    preprocessor=None,
    token_pattern=None     # don't re-tokenise; use our tokens
)
tfidf_mat = tfidf.fit_transform(docs_space).tocsr()   # shape: (n_docs, V)
print("Built TF-IDF matrix with fixed vocabulary.")


# In[9]:


# Embedding features:
# - Unweighted doc vector: mean of token embeddings -> fasttext_unweighted.csv
# - TF-IDF weighted doc vector: weighted mean -> fasttext_tfidf.csv
# Rows align with processed.csv (one vector per review).
assert os.path.exists(FASTTEXT_VEC_PATH), (
    "FastText .vec not found. Unzip wiki-news-300d-1M-subword.vec.zip and set FASTTEXT_VEC_PATH." 
)
ft = KeyedVectors.load_word2vec_format(FASTTEXT_VEC_PATH, binary=False)
ft_dim = ft.vector_size
print(f"Loaded FastText vectors ({FASTTEXT_VEC_PATH}), dim = {ft_dim}")

#Return FastText vector if available; else None
def get_vec(tok):
    return ft[tok] if tok in ft else None 


# In[10]:


# Document Embeddings 
# (a) Unweighted mean of word vectors (bag-of-embeddings)
# (b) TF-IDF weighted mean of word vectors (weighted bag-of-embeddings)

emb_unw   = np.zeros((n_docs, ft_dim), dtype="float32")
emb_tfidf = np.zeros((n_docs, ft_dim), dtype="float32")

# Unweighted mean
for i, toks in tqdm(list(enumerate(docs_tokens)), desc="Unweighted embeddings"):
    toks_v = [t for t in toks if t in v2i]   # stay consistent with vocab
    if not toks_v: 
        continue
    vecs = [get_vec(t) for t in toks_v]
    vecs = [v for v in vecs if v is not None]
    if vecs:
        emb_unw[i] = np.mean(vecs, axis=0)

# TF-IDF weighted mean
vocab_inv = i2v
for i in tqdm(range(n_docs), desc="TF-IDF weighted embeddings"):
    row = tfidf_mat.getrow(i)
    if row.nnz == 0: 
        continue
    idxs, vals = row.indices, row.data
    num = np.zeros(ft_dim, dtype="float32")
    den = 0.0
    for j, idx in enumerate(idxs):
        term = vocab_inv[idx]
        v = get_vec(term)
        if v is None: 
            continue
        w = float(vals[j])
        num += w * v
        den += w
    if den > 0:
        emb_tfidf[i] = num / den


# ### Saving outputs
# Save the BoW count vectors to count_vectors.txt in the required sparse format—one review per line as #row_id, word_index:frequency, built from the Task-1 vocabulary. 

# In[11]:


# code to save output data...
with open(COUNT_VECTORS_TXT, "w", encoding="utf-8") as out:
    for doc_id, toks in enumerate(docs_tokens):
        pairs = doc_sparse_counts(toks)
        out.write(f"#{doc_id},")
        if pairs:
            out.write(",".join(f"{i}:{f}" for i,f in pairs))
        out.write("\n")
print(f"Wrote {COUNT_VECTORS_TXT}") 

np.save(FT_UNW_NPY,   emb_unw)
pd.DataFrame(emb_unw).to_csv(FT_UNW_CSV, index=False)

np.save(FT_TFIDF_NPY, emb_tfidf)
pd.DataFrame(emb_tfidf).to_csv(FT_TFIDF_CSV, index=False)

print("\nArtifacts saved:")
print(f"  - {COUNT_VECTORS_TXT}")
print(f"  - {FT_UNW_NPY}, {FT_UNW_CSV}")
print(f"  - {FT_TFIDF_NPY}, {FT_TFIDF_CSV}") 


# In[12]:


# Quick sanity checks
print("\nSanity check:")
print("First 3 lines of count_vectors.txt:")
with open(COUNT_VECTORS_TXT, "r", encoding="utf-8") as f:
    for _ in range(3):
        line = f.readline()
        if not line: break
        print(line.strip())

if n_docs > 0:
    print("Unweighted emb[0] L2-norm:", np.linalg.norm(emb_unw[0]))
    print("TF-IDF   emb[0] L2-norm:", np.linalg.norm(emb_tfidf[0])) 


# ## Task 3. Clothing Review Classification

# Predict Recommended IND (0/1) using Logistic Regression with 5-fold stratified cross-validation, reporting Accuracy, Precision (macro), Recall (macro), and F1 (macro):
# 
# Q1: Language model comparison — Train/evaluate on each Task-2 feature:
# BoW (from count_vectors.txt), Embedding (Unweighted), and Embedding (TF-IDF Weighted), holding the classifier constant for a fair comparison.
# 
# Q2: Does more information help? — Using BoW, compare Title only, Text only, and Title + Text (concatenated).

# In[13]:


# Code to perform the task...
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# In[14]:


# Load data (label, title, text)
# Task 3 needs labels (Recommended IND) and Title from assignment3.csv,
# plus cleaned Review Text tokens from processed.csv. We align row counts to the minimum to stay safe.
raw = pd.read_csv("assignment3.csv")       # Title, Recommended IND
proc = pd.read_csv("processed.csv")        # tokens (cleaned Review Text)

# Defensive row alignment
n = min(len(raw), len(proc))
raw  = raw.iloc[:n].reset_index(drop=True)
proc = proc.iloc[:n].reset_index(drop=True)

# Label (0/1)
y = raw["Recommended IND"].astype(int).values

# Text sources
title = raw["Title"].fillna("").astype(str) if "Title" in raw.columns else pd.Series([""]*n)
text  = proc["tokens"].fillna("").astype(str)

# Exact token regex required in the brief
TOKEN_PATTERN = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"


# In[15]:



#  Cross validation, metrics, classifier
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}
clf = LogisticRegression(max_iter=2000, n_jobs=None) 


# In[16]:


#BoW with required token pattern; 5-fold CV
def eval_bow(name, series, y_vec):
    X = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True).fit_transform(series)
    res = cross_validate(clf, X, y_vec, cv=cv, scoring=scoring, return_train_score=False)
    return dict(
        experiment=name,
        accuracy_mean=np.mean(res["test_accuracy"]),
        accuracy_std=np.std(res["test_accuracy"]),
        precision_macro_mean=np.mean(res["test_precision_macro"]),
        recall_macro_mean=np.mean(res["test_recall_macro"]),
        f1_macro_mean=np.mean(res["test_f1_macro"]),
    )


# In[17]:


#Load optional embedding matrix (rows=docs, cols=dims). Returns np.ndarray or None.
def try_load_embeddings(path):
    if not os.path.exists(path):
        return None
    arr = pd.read_csv(path, header=None)
    for c in arr.columns:
        arr[c] = pd.to_numeric(arr[c], errors="coerce")
    return arr.fillna(0.0).values 


# In[18]:


#Evaluate a ready numeric/sparse matrix with 5-fold CV
def eval_dense(name, X, y_vec):
    res = cross_validate(clf, X, y_vec, cv=cv, scoring=scoring, return_train_score=False)
    return dict(
        experiment=name,
        accuracy_mean=np.mean(res["test_accuracy"]),
        accuracy_std=np.std(res["test_accuracy"]),
        precision_macro_mean=np.mean(res["test_precision_macro"]),
        recall_macro_mean=np.mean(res["test_recall_macro"]),
        f1_macro_mean=np.mean(res["test_f1_macro"]),
    ) 


# In[19]:


#Reads Task-2 count_vectors.txt with lines like:
#<row_id>, widx:freq,widx:freq,...
#Returns csr_matrix (n_docs x vocab_size_in_file)
    
def load_count_vectors(path):
    if not os.path.exists(path): 
        return None
    rows, cols, data = [], [], []
    max_col, row_idx = -1, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                row_idx += 1
                continue
            # Strip "#<row>," if present
            if s.startswith("#"):
                k = s.find(",")
                s = s[k+1:] if k != -1 else ""
            if s:
                for part in s.split(","):
                    part = part.strip()
                    if ":" not in part: 
                        continue
                    widx, freq = part.split(":", 1)
                    try:
                        c = int(widx); v = float(freq)
                    except:
                        continue
                    rows.append(row_idx); cols.append(c); data.append(v)
                    if c > max_col: max_col = c
            row_idx += 1
    n_docs = row_idx
    n_cols = (max_col + 1) if max_col >= 0 else 0
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, n_cols), dtype=np.float64)

results = [] 


# In[20]:


# -----------------------------
# Q1 — Language model comparisons (same classifier)
#     A) BoW (Review Text)  -> prefer count_vectors.txt
#     B) FastText (Unweighted)       [if file exists]
#     C) FastText (TF-IDF weighted)  [if file exists]
# -----------------------------
# A) BoW via count_vectors.txt  
X_bow = load_count_vectors("count_vectors.txt")
if X_bow is not None:
    # Align rows if needed
    m = min(X_bow.shape[0], len(y))
    X_bow = X_bow[:m]
    y_q1  = y[:m]
    results.append(eval_dense("Q1: BoW (from count_vectors.txt)", X_bow, y_q1))
else:
    # Fallback: build BoW directly from cleaned text using the required regex
    results.append(eval_bow("Q1: BoW (Review Text via CountVectorizer)", text, y))

# B) FastText (Unweighted)
X_unw = try_load_embeddings("fasttext_unweighted.csv")
if X_unw is not None:
    m = min(len(y), X_unw.shape[0])
    results.append(eval_dense("Q1: FastText (Unweighted avg)", X_unw[:m], y[:m]))
    # keep y/text/title aligned with m for subsequent experiments
    y = y[:m]; text = text.iloc[:m]; title = title.iloc[:m]

# C) FastText (TF-IDF weighted)
X_w = try_load_embeddings("fasttext_tfidf.csv")
if X_w is not None:
    m = min(len(y), X_w.shape[0])
    results.append(eval_dense("Q1: FastText (TF-IDF weighted avg)", X_w[:m], y[:m]))
    y = y[:m]; text = text.iloc[:m]; title = title.iloc[:m]


# In[21]:


# -----------------------------
# Q2 — Does more information help? (BoW)
#     - Title only
#     - Text only
#     - Title + Text
# -----------------------------
results.append(eval_bow("Q2: BoW (Title only)", title, y))
results.append(eval_bow("Q2: BoW (Text only)", text, y))
results.append(eval_bow("Q2: BoW (Title + Text)", (title + " " + text).str.strip(), y)) 


# In[22]:


# Save & print
res_df = pd.DataFrame(results).sort_values("experiment")
res_df.to_csv("task3_results.csv", index=False)
print(res_df.to_string(index=False)) 


# ## Summary
# For Task 2, engineered three document representations from the cleaned Review Text: Bag-of-Words count vectors built from the Task-1 vocabulary and saved as count_vectors.txt, unweighted embedding document vectors (mean of word embeddings) saved as fasttext_unweighted.csv, and TF-IDF-weighted embedding vectors saved as fasttext_tfidf.csv. For Task 3, predicted Recommended IND (0/1) using Logistic Regression with 5-fold stratified cross-validation, reporting Accuracy, Precision (macro), Recall (macro), and F1 (macro). first compared the three Task-2 feature types (Q1), then assessed whether adding more information helps by training BoW models on Title only, Text only, and Title + Text (Q2). 
