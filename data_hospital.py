import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def region_to_index():
    df = pd.read_excel("Data_v3_process.xlsx", sheet_name='hospital', usecols=[6],names=None,keep_default_na=False)
    df_li = df.values.tolist()
    data = []

    for s_li in df_li:
            data.append(s_li[0])

    #print(data)

    new = {}
    index = 0

    for i in data:
        if i in new:
            continue
        else:
            new[i]=index
            index += 1

    #print(new)

    data_new = []

    # for i in data:
    #     data_new.append(new[i])

    # print(data_new)

    data_new = [new[i] for i in data]

    #print(data_new)
    return data_new

region_index = region_to_index()

total = []
for col in range(1,6):
    df = pd.read_excel("Data_v3_process.xlsx", sheet_name='hospital', usecols=[col],names=None,keep_default_na=False)
    df_li = df.values.tolist()
    data = []
    result = []
    new = []
    for s_li in df_li:
        if type(s_li[0]) in (int,float):
            data.append(s_li[0])
        else:
            data.append(-1)
    
    count = 0
    for i in data:
        if i != -1:
            count += 1
            result.append(i)
        else:
            continue
    


    mean = np.mean(result)
    #print(mean)

    std = np.std(result,ddof=1)
    #print(std)

    for i in data:
        if i == -1:
            i = mean
            i = (i-mean)/std
        else:
            i = (i-mean)/std
        new.append(i)
    #print(new)
    #print(len(new))
    total.append(new)

total.append(region_index)

#read rating
df_rating = pd.read_excel("Data_v3_process.xlsx", sheet_name='hospital', usecols=[8],names=None,keep_default_na=False)
df_ratingli = df_rating.values.tolist()
data_rating = []

for s_li in df_ratingli:
        data_rating.append(s_li[0])

total.append(data_rating)

data_array = np.array(total)
# print(data_array)
data_final = data_array.T
# print(data_final)
data_list = data_final.tolist()

#delete those with no rating
for row in data_list[:]:
    if row[6]==0:
        data_list.remove(row)

data_final = np.array(data_list)
#print(data_final)

#select 80% of data for training
total_number = len(data_list)
boundary = int(0.8*total_number)

dataset = np.random.permutation(data_final)
data_train = dataset[:boundary]
data_test = dataset[boundary:]
#print(data_train)

x = data_train[ : , 0:6 ]
y = data_train[ : , 6]
# print(x)
# print(y)


#linear regression model
reg = LinearRegression().fit(x, y)

predict_lr = reg.predict(data_test[ : , 0:6])
#print(predict_lr)
true = data_test[ : , 6]
#print(true)
mae_lr = mean_absolute_error(true, predict_lr)
print("linear regression mae:",mae_lr)


#decision tree model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x,(y*10).astype('int32') )

predict_tree = clf.predict(data_test[ : , 0:6 ])
mae_tree = mean_absolute_error(true, predict_tree/10.0)
print("decision tree mae: ",mae_tree)


# logistic regression

# as for result rating, float into int

y=np.squeeze(y)

clf = LogisticRegression(random_state=0).fit(x, (y*10).astype('int32'))
predict_logistic = clf.predict(data_test[ : , 0:6 ])
mae_logistic = mean_absolute_error(true, predict_logistic/10.0)
print("logistic regression mae: ", mae_logistic)