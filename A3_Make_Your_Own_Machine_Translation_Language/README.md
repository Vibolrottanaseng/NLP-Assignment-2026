# A3 - Make Your Own Machine Translation Language (English to Khmer)

In this assignment I chose to create a machine translation for English to Khmer since it is my native language. For this assignment I will explor and comapre between two attention mechanism which are general attention and additive attention. The assignment deliver by implementing a web base  application for demonstration. 

---
## Dataset

Dataset information:

* **Source**: liboaccn/nmt-parallel-corpus (Huggingface)
* **Language Pair**: English to Khmer (EN-KM)
* **Dataset Size**: split by train (2.51M) I cut into 20000 rows for use in this assignment

---
## Text Processing

### **Normaliation**
The preprocessing pipeline applies tokenization but minimal explicit text normalization beyond whitespace trimming.

### **Tokenizationn**

Khmer text is normally written without spaces between words, so word boundaries are not explicit. In this assignment, I used khmer-nltk to perform Khmer word segmentation by automatically splitting Khmer sentences into word tokens using a dictionary/statistical segmentation approach.

1. whitespace normalization
2. tokenization using `khmernltk.word_tokenize` to split khmer sentence into khmer words token
3. Output list of tokens (words), which are then mapped to vocabulary IDs for the model.

**credit:** ([VietHoang1512/khmer-nltk](https://github.com/VietHoang1512/khmer-nltk))

### **Attention Mechanism**

1. **General Attention** <br>
    $e_i = s^T h_i$
1. **Additive Attention** <br>
    $e_i = v_T tanh (W_1h_i + W_2s) $ 



