import numpy as np
from numpy import save, load
from tqdm import tqdm
import nltk
import re
from nltk.corpus import stopwords
from tokenizers import Tokenizer


nltk.download("stopwords")
tokenizer=Tokenizer.from_file("charbpe_007.json")



class Word2vec:

    def __init__(self, corpus=None, w_em2=None, e_x=None, w_em1=None) -> None:
        self.eta = 0.001
        self.dim = 100
        self.w_em2 = w_em2
        self.w_em1 = w_em1
        self.e_x = e_x
        self.stop__ = stopwords.words("english")
        self.corpus = self._sentences()
        self.vocabd, self.vocab_len = self._vocab_gen()
        self.w_i = self._word_to_index()
        self.i_w = self._index_to_word()
        self.v_l = self._vocablist()
        self.o_h_c = self.onehotcode()
        self.td = self._target_context()
        self.w1 = np.random.uniform(-0.8, 0.8, (self.vocab_len, self.dim))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.dim, self.vocab_len))

    def _sentences(self) -> list[list[str]]:
        sentences: list[list[str]] = []
        with open("./datasetexample.txt", "r", encoding="utf-8", errors="ignore") as f:
            stripped = f.read().split(".")
        for i in stripped:
            cleaned_num = re.sub(r"\d", " ", i)
            cleaned = re.sub(r"[^\w\s]", " ", cleaned_num)
            res=tokenizer.encode(cleaned)
            cleaned_=res.tokens
            new=' '.join(cleaned_)
            sentences.append(new.lower().split(' '))
        sentences = [
            list(filter(lambda x: x not in self.stop__, senten)) for senten in sentences
        ]
        return sentences

    def _vocab_gen(self) -> tuple[dict[str, int], int]:
        vocabd = {}
        for senten in self.corpus:
            for word in senten:
                if word not in vocabd:
                    vocabd[word] = 1
                else:
                    vocabd[word] += 1
        vocab_len = len(vocabd)
        return vocabd, vocab_len

    def _word_to_index(self) -> dict[str, int]:
        wordtoindex = {}
        i = 0
        for key in self.vocabd:
            wordtoindex[key] = i
            i += 1
        return wordtoindex

    def _index_to_word(self) -> dict[int, str]:
        indextoword = {}
        i = 0
        for key in self.vocabd:
            indextoword[i] = key
            i += 1
        return indextoword

    def _vocablist(self) -> list[str]:
        vocablist = []
        for key in self.vocabd:
            vocablist.append(key)
        return vocablist

    def onehotcode(self) -> dict[int, list]:
        onehot = {}
        for key, value in self.w_i.items():
            onehot[key] = [0 if value != i else 1 for i in range(len(self.vocabd))]
        return onehot

    def _target_context(self) -> dict[int, list[list[int]]]:
        window = 3
        train_data = {}
        k = 1
        for index, word in enumerate(self.v_l):
            target_corpus = []
            target_corpus.append(self.o_h_c[word])
            for i in range(index - window, index):
                if i >= 0 and i != index:
                    target_corpus.append(self.o_h_c[self.v_l[i]])
            for j in range(index + 1, window + index + 1):
                if j < len(self.v_l):
                    target_corpus.append(self.o_h_c[self.v_l[j]])
            train_data[k] = target_corpus
            del target_corpus
            k += 1
        return train_data

    def forward(self, onehot) -> np.ndarray:
        self.w_em1 = np.dot(self.w1.T, onehot)
        self.w_em2 = np.dot(self.w2.T, self.w_em1)
        return self.w_em2

    def softmax(self, cost) -> np.ndarray:
        e_x = np.exp(cost - np.max(cost))
        return e_x / e_x.sum(axis=0)

    def loss_value(self, value) -> float:
        loss_val = -1 * np.log(value)
        return loss_val

    def backward_pass(self, Error, onehot_t):
        dl_dw2 = np.outer(self.w_em1, Error)
        dl_dw1 = np.outer(onehot_t, (np.dot(self.w2, Error.T)))
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    def model_training(self):
        for k in tqdm(range(1,100)):
            epoch_loss = 0
            for value in self.td.values():
                EI = 0
                loss_values = 0
                onehot_t = value[0]
                context = value[1:]
                for i in range(len(context)):
                    cost = self.forward(onehot_t)
                    y_pred = self.softmax(cost)
                    y_actual = context[i]
                    EI_TEMP = np.array(y_pred) - np.array(y_actual)
                    EI += EI_TEMP
                    for i in range(len(y_actual)):
                        if y_actual[i] == 1:
                            values = y_pred[i]
                            break
                    loss_value_temp = self.loss_value(values)
                    loss_values += loss_value_temp
                epoch_loss = epoch_loss + loss_values
                self.backward_pass(EI, onehot_t)
            print(f"epoch no {k} loss is {epoch_loss}")

    def savemodel(self):
        save("embedding_matrix.npy", self.w1)
        save("training_weight.npy", self.w2)

    def loadmodel(self):
        self.w1 = load("embedding_matrix.npy")
        self.w2 = load("training_weight.npy")


def main():
    w2v = Word2vec()
    w2v.model_training()
    w2v.savemodel()

if __name__ == "__main__":
    main()
