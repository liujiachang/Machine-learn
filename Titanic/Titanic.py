import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
#from scipy import stats
# 读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# 查看数据
#train_data.info()

# 选择特征
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train = train_data[features]
test = test_data[features]
y = train_data['Survived']
#train.info()
# 填充缺失值
#age = stats.mode(train['Age'])[0][0]
#print(age)
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S',inplace=True)

# 特征转换,编码
devc = DictVectorizer(sparse = False)
#print(train)
train = devc.fit_transform(train.to_dict(orient='record'))
test = devc.fit_transform(test.to_dict(orient='record'))

#print(devc.feature_names_)
#print(type(y))
# 逻辑回归模型
lr = LogisticRegression(solver='lbfgs')
# 十折交叉验证正确率的平均值
print(np.mean(cross_val_score(lr,train,y,cv=10)))
#训练
lr.fit(train,y)
# 预测
result = lr.predict(test)
# 输出csv文件
testy = {'PassengerId': test_data['PassengerId'],'Survived':result}
testy = pd.DataFrame(testy)
testy.to_csv('testy.csv',index=False)
