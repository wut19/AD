from gensim.models import word2vec
from train_AAE import train

model = word2vec.Word2Vec.load("word2vec.model")
train(model)