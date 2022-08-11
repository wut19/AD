import pickle


def str2onehot(data_path):
    input = open(data_path,'rb')
    data = pickle.load(input)
    
    true_rv = []
    fake_rv = []
    for class1,items in data.items():
        if class1 == 1:
            true_rv += [(class1,item) for item in items]
        else :
            fake_rv += [(class1,item) for item in items]
    
    max_len = 0
    for label,review in true_rv+fake_rv:
        if len(review)>max_len:
            max_len = len(review)          
    print(max_len)
    
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
    output = open('./dataset/onehot_data.pkl','wb')
    pickle.dump((true_onehot,fake_onehot), output)
    output.close()
    print(len(dic))

if __name__=="__main__":
    data_path='./dataset/textdata.pkl'
    str2onehot(data_path)
    

    