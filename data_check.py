## label,data examination
import pickle
# no problems
data_path = './dataset/textdata.pkl'
input = open(data_path,'rb')
data = pickle.load(input)
true_rv = []
fake_rv = []
for class1,items in data.items():
    if class1 == 1:
        true_rv += [(class1,item) for item in items]
    else :
        fake_rv += [(class1,item) for item in items]
# textdata
print("true_first10",true_rv[:10])
print("fake_first10",fake_rv[:10])
print("true_last10",true_rv[-10:-1])
print("fake_last10",fake_rv[-10:-1])

max_len = 766

# no problem
dic = {}
true_onehot = []
for label,review in true_rv:
    review_onehot = []
    for word in review:
        if word not in dic.keys():
            dic[word] = len(dic)+1
        review_onehot.append(dic[word])
    while len(review_onehot)<max_len:
        review_onehot.append(0)
    true_onehot.append((label,review_onehot))

fake_onehot = []
for label,review in fake_rv:
    review_onehot = []
    for word in review:
        if word not in dic.keys():
            dic[word] = len(dic)+1
        review_onehot.append(dic[word])
    while len(review_onehot)<max_len:
        review_onehot.append(0)
    fake_onehot.append((label,review_onehot))

output = open('./dataset/dic.txt','w')
for key,value in dic.items():
    output.write("%s:%d"%(key,value))
output.close()

# onehot data
print(true_onehot[0],true_onehot[10],true_onehot[-10],true_onehot[-2])
print(fake_onehot[0],fake_onehot[10],fake_onehot[-10],fake_onehot[-2])

