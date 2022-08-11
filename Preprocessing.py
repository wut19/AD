import re
import nltk  
#nltk.download('punkt') 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import pickle

file_name = './dataset/YelpNYC/metadata'
file_object = open(file_name)
data = file_object.read()
data = data.split()
label = data[3::5]
label = [int(x) for x in label]
file_object.close()

file_name = './dataset/YelpNYC/reviewContent'
file_object = open(file_name,encoding='utf-8')
data = file_object.read()
data = re.split('[\t\n]',data)
review = data[3::4]
file_object.close()

ps = PorterStemmer()
class_bin = {}
class_bin[1] = []
class_bin[-1] = []
for i in range(len(review)):
    review_clean = re.sub(r'[^a-zA-Z]', ' ', review[i])
    reveiw_tokenized = word_tokenize(review_clean)
    review_with_no_stopwords = [word for word in reveiw_tokenized if word not in stopwords.words('english')] 
    class_bin[label[i]] += [review_with_no_stopwords]
    # print(review_with_no_stopwords)
    
output = open('./dataset/textdata.pkl','wb')
pickle.dump(class_bin, output)
output.close()




