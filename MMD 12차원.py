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
data_real2=df.iloc[:4704,[1,2,3,4,5,6,7,8,9,10,14,15]]
data_real2

#정규화하기
np_mean2=np.mean(data_real2,axis=0)

print(np_mean2)
np_std2=np.std(data_real2,axis=0)
print(np_std2)
normal_data2=np.divide(np.subtract(data_real2,np_mean2),np_std2)
normal_data2.head(4704)


#svd
from numpy.linalg import svd
# 원본 행렬을 출력하고, U, Sigma, Vt 의 차원 확인
print('원본 행렬:\n',normal_data2)
U2, Sigma2, Vt2 = svd(normal_data2)
print(U2.shape, Sigma2.shape,Vt2.shape)
print(Sigma2)
print(U2)
print(Vt2)

#우특이벡터 V구하기
V2=np.transpose(Vt2)
V2.shape
print(V2)
print('V행렬 : ',V2)

#선형변환된 데이터셋 구하기
hat_X2=np.dot(normal_data2,V2)
print(hat_X2)
hat_X2.shape

#뮤 구하기
mu2=np.mean(hat_X2)
print(mu2)
vector_mu2=np.dot(np.ones((4704,1) ),mu2)
vector_mu2.shape


#Calculate Mahalanobis distance
x_minus_mu2=hat_X2-vector_mu2
x_minus_mu2.shape
print(x_minus_mu2)
x_minus_mu_T2=x_minus_mu2.T
x_minus_mu_T2.shape

inv_cov2=np.dot(x_minus_mu_T2,x_minus_mu2)/4704
print(inv_cov2)
left_term2=np.dot(x_minus_mu2,inv_cov2)
print(left_term2)
left_term2.shape
x_minus_mu_T2.shape
A2=np.dot(left_term2,x_minus_mu_T2)
A2.shape
print(A2)
sourceNDArray2 = np.array(A2)
targetNDArray2 = np.diag(sourceNDArray2)
print(targetNDArray2)
targetNDArray2.shape

f_MMD2=np.sqrt(targetNDArray2)
f_MMD2.shape
print(f_MMD2)
f_MMD2 = pd.DataFrame(f_MMD2) #데이터 프레임으로 전환
f_MMD2.to_excel(excel_writer='f_mmd_normal2.xlsx') #엑셀로 저장


#비정상데이터로 다시 추출
df_ab=pd.DataFrame(data)
data_abnormal2=df_ab.iloc[4704:7846,[1,2,3,4,5,7,8,9,10,14,15]]
data_abnormal2



#정규화하기
np_mean_abnormal2=np.mean(data_abnormal2,axis=0)
print(np_mean_abnormal2)
np_std_abnormal2=np.std(data_abnormal2,axis=0)
print(np_std_abnormal2)
abnormal_data2=np.divide(np.subtract(data_abnormal2,np_mean_abnormal2),np_std_abnormal2)
abnormal_data2.head(3141)

abnormal_data2 = pd.DataFrame(abnormal_data2)
abnormal_data2.insert(5,'tcp_urg_packet',0)
abnormal_data2


#svd
from numpy.linalg import svd
# 원본 행렬을 출력하고, U, Sigma, Vt 의 차원 확인
print('원본 행렬:\n',normal_data2)
U2, Sigma2, Vt2 = svd(normal_data2)
print(U2.shape, Sigma2.shape,Vt2.shape)
print(Sigma2)
print(U2)
print(Vt2)

#우특이벡터 V구하기
V2=np.transpose(Vt2)
V2.shape
print(V2)
print('V행렬 : ',V2)

#선형변환된 데이터셋 구하기
hat_X_abnormal2=np.dot(abnormal_data2,V2)
print(hat_X_abnormal2)
hat_X_abnormal2.shape

#뮤 구하기
mu_abnormal2=np.mean(hat_X_abnormal2)
print(mu_abnormal2)
vector_mu_abnormal2=np.dot(np.ones((3141,1) ),mu2)
vector_mu_abnormal2.shape

#Calculate Mahalanobis distance
x_minus_mu_abnormal2=hat_X_abnormal2-vector_mu_abnormal2
x_minus_mu_abnormal2.shape
print(x_minus_mu_abnormal2)
x_minus_mu_T_abnormal2=x_minus_mu_abnormal2.T
x_minus_mu_T_abnormal2.shape

inv_cov_abnormal2=np.dot(x_minus_mu_T_abnormal2,x_minus_mu_abnormal2)/3141
print(inv_cov_abnormal2)
left_term_abnormal2=np.dot(x_minus_mu_abnormal2,inv_cov_abnormal2)
print(left_term_abnormal2)
left_term_abnormal2.shape
x_minus_mu_T_abnormal2.shape
F_MMD_abnormal2=np.dot(left_term_abnormal2,x_minus_mu_T_abnormal2)
F_MMD_abnormal2.shape
print(F_MMD_abnormal2)

#대각행렬만 추출
sourceNDArray_abnormal2 = np.array(F_MMD_abnormal2)
targetNDArray_abnormal2 = np.diag(sourceNDArray_abnormal2)
print(targetNDArray_abnormal2)
targetNDArray_abnormal2.shape

#비정상데이터의 f_mmd
f_mmd_abnormal2=np.sqrt(targetNDArray_abnormal2)
f_mmd_abnormal2.shape
print(f_mmd_abnormal2)
f_mmd_abnormal2 = pd.DataFrame(f_mmd_abnormal2) #데이터 프레임으로 전환
f_mmd_abnormal2.to_excel(excel_writer='f_mmd_abnormal2.xlsx') #엑셀로 저장


f_MMD2
b = np.where(f_MMD2<2.266,True,False)
num_ones3 = (b == 1).sum()
print(num_ones3)


f_mmd_abnormal2
c=np.where(f_mmd_abnormal2>2.266,True,False)
num_ones4 = (c == 1).sum()
print(num_ones4)


#분류 성능 구하기

# Precision 정밀도
from sklearn.metrics import precision_score
df=pd.DataFrame(data)
data3=df.iloc[:,[16]]
y_true=np.where(data3=='benign',True,False)
d=np.where(f_MMD2<2.266,True,False)
e=np.where(f_mmd_abnormal2>2.266,True,False)
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
F=np.array([['F1_score','Recall','Accuracy','Specificity','Precision','MCC','Fall out'],[F1_score,Recall,Accuracy,Specificity,Precision,MCC,Fall_out]])
print(F)
F = pd.DataFrame(F) #데이터 프레임으로 전환
F.to_excel(excel_writer='MMD12차원.xlsx') #엑셀로 저장'
