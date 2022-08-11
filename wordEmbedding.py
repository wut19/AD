from gensim.models import word2vec
from gensim.models import keyedvectors
import Review_dataload as Rd

# # 引入数据集
# raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# # 切分词汇
# sentences= [s.split() for s in raw_sentences]
# # print(sentences)
# # 构建模型
# model = word2vec.Word2Vec(sentences, min_count=1)

# # 进行相关性比较
# # model.similarity('dogs','you')

# print(model.wv.similarity('dogs', 'fox'))

def wordEmbedding(data_path,min_count=10,size=200,workers=1,epochs=30):
    train_set,_,_ = Rd.make_datasets(data_path)
    model = word2vec.Word2Vec(sentences=train_set.reviews,min_count=min_count,vector_size=size,workers=workers,epochs=epochs)
    model.save("word2vec.model")

if __name__=="__main__":
    data_path='./dataset/textdata.pkl'
    wordEmbedding(data_path)
    print('done!')