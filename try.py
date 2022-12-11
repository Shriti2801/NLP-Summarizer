import pandas as pd
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

print("Loading Data...")
df = pd.read_csv("tennis.csv")
# print(df.head())

print("Tokenizing articles into sentences...")
article_list = []
for article in df['article_text']:
    article_list.append(sent_tokenize(article))
  

article_list_enum = []
sent_num=0
for i in range(len(article_list)):
    for sent in article_list[i]:
        article_list_enum.append((sent_num,i,sent))
        sent_num+=1
print(article_list_enum[50])
sentences = [s for article in article_list for s in article]

word_embeddings = {}
print("Loading Glove...")
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

print("Cleaning Data...")
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
print("Removing stop words...")
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]   

# print(sentences[0])
# print(clean_sentences[0])


# temp_l = []
# sen0_list = clean_sentences[0].split()
# for word in sen0_list:
#     temp_l.append(word_embeddings.get(word, np.zeros((100,))))
# print(temp_l)
# print(sum(temp_l))
# v = sum(temp_l)/(len(sen0_list)+0.001)
# print(v)
# print(np.zeros((100,)))

print('Generating sentence vectors...')
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

print('Generating Cosine similarity matrix...')
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
print(scores)
# # sentences = enumerate(sentences)    
# print(article_list_enum)
rs = ((scores[sent_num],article_id,s) for sent_num,article_id,s in article_list_enum)
ranked_sentences = sorted(rs, reverse=True)
def get_summery(article_id):
    for sent in ranked_sentences:
        summary = ""
        # print(sent)
        if(article_id == sent[1]): # sent[1] hold article id 
            summary = sent[2]
            break
    return summary 
    

for i in range(5):
    print("Article: \n")
    print(df['article_text'][i])
    print("\n")

    print("Summary: \n")
    print(get_summery(i))
    print("\n")