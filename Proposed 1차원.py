import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from numpy import *

#데이터 불러오기
data=pd.read_csv("C:/Users/mcnl/PycharmProjects/test/android_traffic.csv",delimiter=';')
data

#string type 데이터 제외시키기
df=pd.DataFrame(data)
data_real=df.iloc[:4704,[1,2,3,4,5,6,7,8,9,10,14,15]]
data_real

#정규화하기
np_mean=np.mean(data_real,axis=0)
print(np_mean)
np_std=np.std(data_real,axis=0)
print(np_std)
normal_data=np.divide(np.subtract(data_real,np_mean),np_std)
normal_data.head(4704)


#svd
from scipy.sparse.linalg import svds
# 원본 행렬을 출력하고, U, Sigma, Vt 의 차원 확인
print('원본 행렬:\n',normal_data)
num_components=1
U, Sigma, Vt = svds(normal_data,k=num_components)
print(U.shape, Sigma.shape,Vt.shape)
print(Sigma)
print(U)
print(Vt)

#우특이벡터 V구하기
V=np.transpose(Vt)
V.shape
print(V)
print('V행렬 : ',V)

#선형변환된 데이터셋 구하기
hat_X=np.dot(normal_data,V)
print(hat_X)
hat_X.shape

#뮤 구하기
mu=np.mean(hat_X)
print(mu)
vector_mu=np.dot(np.ones((4704,1)),mu)
vector_mu.shape


#Calculate Mahalanobis distance
x_minus_mu=hat_X-vector_mu
x_minus_mu.shape
print(x_minus_mu)
x_minus_mu_T=x_minus_mu.T
x_minus_mu_T.shape

inv_cov=np.dot(x_minus_mu_T,x_minus_mu)/4704
print(inv_cov)
left_term=np.dot(x_minus_mu,inv_cov)
print(left_term)
left_term.shape
x_minus_mu_T.shape
F_MMD=np.dot(left_term,x_minus_mu_T)
F_MMD.shape
print(F_MMD)

#대각행렬만 추출
sourceNDArray = np.array(F_MMD)
targetNDArray = np.diag(sourceNDArray)
print(targetNDArray)
targetNDArray.shape

#정상데이터의 f_mmd
import pandas as pd
f_mmd=np.sqrt(targetNDArray)
f_mmd.shape
print(f_mmd)
f_mmd = pd.DataFrame(f_mmd) #데이터 프레임으로 전환
f_mmd.to_excel(excel_writer='f_mmd_normal.xlsx') #엑셀로 저장

#비정상데이터로 다시 추출
df_ab=pd.DataFrame(data)
data_abnormal=df_ab.iloc[4704:7846,[1,2,3,4,5,7,8,9,10,14,15]]
data_abnormal



#정규화하기
np_mean_abnormal=np.mean(data_abnormal,axis=0)
print(np_mean_abnormal)
np_std_abnormal=np.std(data_abnormal,axis=0)
print(np_std_abnormal)
abnormal_data=np.divide(np.subtract(data_abnormal,np_mean_abnormal),np_std_abnormal)
abnormal_data.head(3141)

abnormal_data2 = pd.DataFrame(abnormal_data)
abnormal_data2.insert(5,'tcp_urg_packet',0)
abnormal_data2



#svd
from scipy.sparse.linalg import svds
# 원본 행렬을 출력하고, U, Sigma, Vt 의 차원 확인
print('원본 행렬:\n',abnormal_data2)
num_components=1
U, Sigma, Vt = svds(abnormal_data2,k=num_components)
print(U.shape, Sigma.shape,Vt.shape)
print(Sigma)
print(U)
print(Vt)

#우특이벡터 V구하기
V=np.transpose(Vt)
V.shape
print(V)
print('V행렬 : ',V)

#선형변환된 데이터셋 구하기
hat_X_abnormal=np.dot(abnormal_data2,V)
print(hat_X_abnormal)
hat_X_abnormal.shape

#뮤 구하기
mu_abnormal=np.mean(hat_X_abnormal)
print(mu_abnormal)
vector_mu_abnormal=np.dot(np.ones((3141,1) ),mu)
vector_mu_abnormal.shape

#Calculate Mahalanobis distance
x_minus_mu_abnormal=hat_X_abnormal-vector_mu_abnormal
x_minus_mu_abnormal.shape
print(x_minus_mu_abnormal)
x_minus_mu_T_abnormal=x_minus_mu_abnormal.T
x_minus_mu_T_abnormal.shape

inv_cov_abnormal=np.dot(x_minus_mu_T_abnormal,x_minus_mu_abnormal)/3141
print(inv_cov_abnormal)
left_term_abnormal=np.dot(x_minus_mu_abnormal,inv_cov_abnormal)
print(left_term_abnormal)
left_term_abnormal.shape
x_minus_mu_T_abnormal.shape
F_MMD_abnormal=np.dot(left_term_abnormal,x_minus_mu_T_abnormal)
F_MMD_abnormal.shape
print(F_MMD_abnormal)

#대각행렬만 추출
sourceNDArray_abnormal = np.array(F_MMD_abnormal)
targetNDArray_abnormal = np.diag(sourceNDArray_abnormal)
print(targetNDArray_abnormal)
targetNDArray_abnormal.shape

#비정상데이터의 f_mmd
f_mmd_abnormal=np.sqrt(targetNDArray_abnormal)
f_mmd_abnormal.shape
print(f_mmd_abnormal)
f_mmd_abnormal = pd.DataFrame(f_mmd_abnormal) #데이터 프레임으로 전환
f_mmd_abnormal.to_excel(excel_writer='f_mmd_abnormal.xlsx') #엑셀로 저장


f_mmd
b = np.where(f_mmd<2.266,True,False)
num_ones1 = (b == 1).sum()
print(num_ones1)

f_mmd_abnormal
c=np.where(f_mmd_abnormal>2.266,True,False)
num_ones2 = (c == 1).sum()
print(num_ones2)


#분류 성능 구하기

# Precision 정밀도
from sklearn.metrics import precision_score
df=pd.DataFrame(data)
data3=df.iloc[:,[16]]
y_true=np.where(data3=='benign',True,False)
d=np.where(f_mmd<2.266,True,False)
e=np.where(f_mmd_abnormal>2.266,True,False)
y_pred=np.concatenate((d, e), axis=0)
Precision=precision_score(y_true, y_pred, pos_label=1)

# Accuracy 정확도
from sklearn.metrics import accuracy_score
Accuracy=accuracy_score(y_true, y_pred)

# recall (재현율)
from sklearn.metrics import recall_score
Recall=recall_score(y_true, y_pred, pos_label=1)

# f1 score
from sklearn.metrics import f1_score
F1_score=f1_score(y_true, y_pred, pos_label=1)

#Fall out
E=np.array(e).flatten()
num_ones = (E == 1).sum()
Fall_out=num_ones/len(e)

#Specificity
Specificity=1-Fall_out

#MCC
from sklearn.metrics import matthews_corrcoef
MCC=matthews_corrcoef(y_true,y_pred)


#성능지표 출력
A=np.array([['F1_score','Recall','Accuracy','Specificity','Precision','MCC','Fall out'],[F1_score,Recall,Accuracy,Specificity,Precision,MCC,Fall_out]])
print(A)
A = pd.DataFrame(A) #데이터 프레임으로 전환
A.to_excel(excel_writer='Proposed1차원.xlsx') #엑셀로 저장'