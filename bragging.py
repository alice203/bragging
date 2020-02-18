import pandas as pd
import numpy as np
import nltk
from numpy import savetxt

# Load data
data = pd.read_excel("blurb.xlsx")
#print(data.head(15))
#print(all_data.dtypes)
#print(len(all_data))


# Seperate data 
all_data_matrix = np.array(data)
#print(all_data_matrix.shape)
#print(all_data_matrix[0].dtype)

headlines = all_data_matrix[:,:2]
#print(headlines.shape)
#print(headlines[50:100])

blurbs = all_data_matrix[:,::2]
#print(blurbs.shape)
#print(blurbs[:10])


# Clean data
def detect_obs_nan(X):
    for i, array in enumerate(X):        
        clean = array[1]        
        if isinstance(clean, float):
            print(i)

detect_obs_nan(headlines)

#Careful!!!! Only run this once!
headlines = np.delete(headlines, 86737, axis =0)
headlines = np.delete(headlines, 109565, axis =0)
headlines = np.delete(headlines, 259745, axis =0)

def clean(X):
    new_X = []
    for i, array in enumerate(X):
        new_array = []
        new_array.append(array[0])
        #print(i)
        clean = array[1] 
        clean = clean.lower()
        clean = clean.replace("world's", "worlds")
        clean = clean.replace(".", " ")
        clean = clean.replace("1", "")
        clean = clean.replace("2", "")
        clean = clean.replace("3", "")
        clean = clean.replace("4", "")
        clean = clean.replace("5", "")
        clean = clean.replace("6", "")
        clean = clean.replace("7", "")
        clean = clean.replace("8", "")
        clean = clean.replace("9", "")
        clean = clean.replace("0", "")    
        clean = clean.replace(",", " ")
        clean = clean.replace(".", " ")
        clean = clean.replace("!", " ")
        clean = clean.replace("?", " ")
        clean = clean.replace(";", " ")
        clean = clean.replace(":", " ")
        clean = clean.replace('"', " ")
        clean = clean.replace('—', " ")
        clean = clean.replace('-', " ")
        clean = clean.replace('(', " ")
        clean = clean.replace(')', " ")
        clean = clean.replace('/', " ")
        clean = clean.replace('|', " ")
        clean = clean.replace('♫♫', " ")
        clean = clean.replace('&', " ")
        clean = clean.replace('%', " ")
        clean = clean.replace('$', " ")
        clean = clean.replace('€', " ")
        clean = clean.replace('’', " ")
        clean = clean.replace("'", " ")
        clean = clean.replace('´', " ") 
        clean = clean.replace('â', " ")
        clean = clean.replace('¢', " ")
        clean = clean.replace("+", " ")
        clean = clean.replace("#", " ")
        clean = clean.replace("@", " ")
        clean = clean.replace('*', " ")
        new_array.append(clean)
        a = np.array(new_array)
        new_X.append(a)
    return np.array(new_X)

headlines = clean(headlines)

#print(headlines.shape)
#print(headlines[10])

#Remove words with less than 4 characters
def remove_short_words(X):
    new_X = []
    for array in X:
        new_array = []
        new_array.append(array[0])       
        lw = array[1]
        lw = lw.split(" ") 
        lw_new = []
        for word in lw:
            if len(word) < 3:
                continue
            else:
                lw_new.append(word)
        string = " ".join(lw_new)
        new_array.append(string)
        a = np.array(new_array)
        new_X.append(a)        
    return np.array(new_X)

headlines_2 = remove_short_words(headlines)

#print(headlines_2[80:100])


#Tokenization
## test
c = "the worlds warmest neck gaiter with stash pocket"
tokens = nltk.word_tokenize(c)
d = nltk.pos_tag(tokens)
print(d)
#print(d[0][1])

def tokenization(X):
    new_X = []
    for array in X:
        new_array = []
        new_array.append(array[0]) 
        prx = array[1]
        tl = nltk.word_tokenize(prx)
        tl = nltk.pos_tag(tl)
        new_array.append(tl)
        a = np.array(new_array)
        new_X.append(a)     
    return np.array(new_X)

headlines_3 = tokenization(headlines_2)

#print(headlines_3[80:100])

def create_columns_type(X, word_type):
    col = []
    for i, row in enumerate(X):
        proxy = ""
        for word in row[1]:
            if word[1] == word_type:
                if len(proxy) > 0:
                    proxy += ", " 
                    proxy += word[0]
                else:  
                    proxy += word[0]
            else:
                continue
        col.append(proxy)           
    return np.array(col)

adj = create_columns_type(headlines_3, 'JJ').reshape(408634,1)
#print(adj[8])
#print(adj.shape)

com = create_columns_type(headlines_3, 'JJR').reshape(408634,1)
#print(com.shape)

sup = create_columns_type(headlines_3, 'JJS').reshape(408634,1)
#print(sup.shape)

def create_columns_word(X, word_type, special):
    col = []
    for i, row in enumerate(X):
        proxy = ""
        for word in row[1]:
            if word[1] == word_type and word[0] == special :
                #print('yeah')
                proxy += "1"
            else:
                continue
        col.append(proxy)            
    return np.array(col)

worlds = create_columns_word(headlines_3, 'NNS', 'worlds').reshape(408634,1)
most = create_columns_word(headlines_3, 'NNS', 'most').reshape(408634,1)
first = create_columns_word(headlines_3, 'NNS', 'most').reshape(408634,1)

#Concatenate columns 

#print(headlines_3.shape)
headlines_4 = np.concatenate((headlines, adj), axis = 1)
#print(headlines_4.shape)

headlines_5 = np.concatenate((headlines_4, com), axis = 1)

headlines_6 = np.concatenate((headlines_5, sup), axis = 1)

headlines_7 = np.concatenate((headlines_6, most), axis = 1)

headlines_8 = np.concatenate((headlines_7, first), axis = 1)

headlines_9 = np.concatenate((headlines_8, worlds ), axis = 1)

#print(headlines_9.shape)

# Create new data frame from array
r = pd.DataFrame(headlines_9, columns = ['index', 'headline', 'adjective', 'comparative', 'superlative', 'most', 'first', 'worlds'])

np.savetxt('bragging.csv', r, delimiter=',', fmt='%s')