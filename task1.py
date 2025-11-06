#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
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
# * nltk
# 
# 
# ## Introduction
# This task prepares the raw clothing‐review data for downstream modelling by converting unstructured text into a clean, consistent, and reproducible representation. We work with ~19.6k reviews and focus only on the “Review Text” field (the title is ignored in Task 1). The goal is to remove noise, standardise tokens, remove stopwords, Prunes rare and overly common terms and build a vocabulary that accurately reflects meaningful language use in the corpus so that Task 2 (feature construction) and Task 3 (classification) are built on a solid foundation.
# 
# Outputs:
# 
# * processed.csv — cleaned tokens per review, ready for vectorisation 
# 
# * vocab.txt — an alphabetically sorted unigram vocabulary in the format word:index starting from 0, which is the key for interpreting sparse encodings in later tasks.
# 
#  

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment 
import pandas as pd 
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter


# ### 1.1 Examining and loading data
# We load the provided dataset from assignment3.csv, which contains the Title, Review Text, and the target label Recommended IND (0 = not recommended, 1 = recommended). To ensure the text is ready for processing, the Review Text field is coerced to string and missing values are filled with empty strings. As a quick sanity check, we confirm the total number of rows, look for missing values in critical columns, examine basic length statistics of the review text, and inspect the class balance of the target label. These checks help surface common issues (for example, nulls, empty reviews, severe class imbalance) before we proceed to tokenisation and vocabulary construction in Task 1.

# In[2]:


# Code to inspect the provided data file 
# Load raw data and make sure critical columns exist; coerce Review Text to string
data_csv_path = "assignment3.csv"    
df = pd.read_csv(data_csv_path)

# Ensure Review Text is string
df["Review Text"] = df["Review Text"].fillna("").astype(str)

#display the length of reviews 
print("Number of reviews:", len(df))
df.head()


# ### 1.2 Pre-processing data
# In text preprocessing, convert raw Review Text into a clean, consistent form suitable for modelling. Text is tokenised with the required regex ([a-zA-Z]+(?:[-'][a-zA-Z]+)?) to keep valid words (including hyphen/apostrophe forms), then lower-cased and very short tokens are removed to reduce noise. We remove stopwords (from stopwords_en.txt) so common function words don’t dominate. Next, we prune rare terms that appear only once across the corpus and drop the top-20 highest document-frequency words to improve discrimination. The result is a compact, informative vocabulary and cleaned tokens saved to processed.csv and vocab.txt, providing a reproducible foundation for Task 2 features and Task 3 classification.

# In[3]:


# code to perform the task...
# Tokenization — keep alphabetic tokens and common hyphen/apostrophe forms as per spec
tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def tokenize(text):
    return tokenizer.tokenize(text)

# Apply to dataset
df["tokens"] = df["Review Text"].apply(tokenize)
df[["Review Text", "tokens"]].head()


# In[4]:


# Load stopwords from the provided file (stopwords_en.txt) into a lowercase set
stopwords_set = set()
with open("stopwords_en.txt", "r", encoding="utf-8") as f:
    for line in f:
        w = line.strip()
        if w:
            stopwords_set.add(w.lower())
print("Stopwords loaded:", len(stopwords_set))

 

# Normalize and remove very short tokens
# single-character tokens are mostly noise, lowercasing reduces case variants.
def clean_tokens(tokens):
    cleaned = []
    for t in tokens:
        t = t.lower()
        if len(t) == 1:     # remove only single-character words
            continue
        if t in stopwords_set:  # now 'stopwords' is your set
            continue
        cleaned.append(t)
    return cleaned

# Apply
df["tokens"] = df["tokens"].apply(clean_tokens)
df["tokens"].head()


# In[5]:


#Remove stopwords — keeps content words for downstream modelling 
def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords_set]

df["tokens"] = df["tokens"].apply(remove_stopwords)
df["tokens"].head()


# In[6]:



# Rare-term pruning (TF=1): remove terms that appear only once in the entire corpus
# This uses term frequency (total occurrences), not document frequency.
 
all_tokens = [t for doc in df["tokens"] for t in doc]
freq = nltk.FreqDist(all_tokens)

# Words appearing only once
singleton_words = {w for w, c in freq.items() if c == 1}

def remove_singletons(tokens):
    return [t for t in tokens if t not in singleton_words]


df["tokens"] = df["tokens"].apply(remove_singletons)
print("Vocabulary size after removing singletons:", len(set([t for doc in df["tokens"] for t in doc])))
  


# In[7]:



# High-DF pruning: remove the 20 most frequent words by Document Frequency (DF)
# DF counts in how many documents a word appears, use set(doc) to avoid double-counting.

df_counter = Counter()
for doc in df["tokens"]:
   df_counter.update(set(doc))

top20 = [w for w, _ in df_counter.most_common(20)]
top20_set = set(top20)

def remove_top20(tokens):
   return [t for t in tokens if t not in top20_set]

df["tokens"] = df["tokens"].apply(remove_top20)

print("Top-20 removed:", top20)


# ## Saving required outputs
# export the cleaned tokens (space-joined) to processed.csv, and write an alphabetically sorted unigram vocabulary with indices starting at 0 to vocab.txt for consistent word–index mapping in later tasks.

# In[8]:


# code to save output data...
 
# Save processed.csv — one review per line, tokens space-joined
 
processed_texts = df["tokens"].apply(lambda x: " ".join(x))
processed_texts.to_csv("processed.csv", index=False,header='Review Text')
print("Saved processed.csv")


# In[9]:



# Build vocab.txt — unique tokens, alphabetically sorted, with indices starting at 0 

vocab = sorted(set([t for doc in df["tokens"] for t in doc]))
with open("vocab.txt", "w", encoding="utf-8") as f:
   for idx, w in enumerate(vocab):
       f.write(f"{w}:{idx}\n")

print("Vocabulary size:", len(vocab))
print("Saved vocab.txt")


# In[10]:



# Quick Sanity Check

print("First 5 processed reviews:")
print(processed_texts.head())

print("\nFirst 10 vocab entries:")
print(vocab[:10])


# ## Summary
#  
# In Task 1, transformed the raw clothing reviews into a clean, model-ready format. Using the required tokeniser, lower-cased text, removed very short tokens, filtered stopwords, and pruned both rare (TF=1) and overly common (top-20 DF) terms. The outputs are **processed.csv** (cleaned tokens per review) and **vocab.txt** (alphabetically sorted **word:index** mapping starting at 0). This pipeline reduces noise and sparsity, preserves meaningful words, and provides a reproducible foundation for Task 2 feature construction and Task 3 classification.
# 
