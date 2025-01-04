import numpy as np
from numpy import load
from numpy.linalg import norm   
from word_to_vec import Word2vec



w2v = Word2vec()
w2v.loadmodel()
embedding_machine = w2v.w1
vocabs = w2v._vocablist()
embedding_dict = {i: j for i, j in zip(vocabs, embedding_machine)}
with open(".\embedding_dict.txt", "w",encoding="utf-8") as f:
    for i in vocabs:
        f.write(f"{i} --> {embedding_dict[i]}\n")
def cosinesimilarity(word:str)->list[float]:
    cosines={}
    for key in embedding_dict:
        dot_product=np.dot(embedding_dict[word],embedding_dict[key])
        norm_product=norm(embedding_dict[word])*norm(embedding_dict[key])
        cosine_similarity=dot_product/norm_product
        cosines[key]=cosine_similarity
    return cosines
    
cosine_simil=cosinesimilarity('ġ'+'bible')

sorted_dict={key:value for key,value in sorted(cosine_simil.items(),key=lambda item:item[1])}
print(list(sorted_dict.keys())[:-10:-1])



    

