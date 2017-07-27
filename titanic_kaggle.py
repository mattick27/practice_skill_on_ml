import pandas as df
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

x = df.read_csv('train.csv')
x = x.sort_values(by = "Survived")
final = df.read_csv('test.csv')

#---------------------------------------#

final_1 = np.array(final)[:,[3,1]]
final_sex = np.array(final_1)[:,[0]]
final_pclass = np.array(final_1)[:,[1]]

for i in range (len(final_sex)):
    if  final_sex[i]== "male":
         final_sex[i]= 1
    else:
         final_sex[i]= 0

final_sex = final_sex * 5
final_pclass += 10
final_sum = np.add(final_sex,final_pclass)
#print(final_sum)


#---------------------------------------#
x_test_1 = np.array(x)[:222,[2,4]]
x_test_2 = np.array(x)[554:690,[2,4]]

x_test = list(x_test_1)+list(x_test_2)
#print(len(x_test))
for i in range (len(x_test)):
    if  x_test[i][1]== "male":
         x_test[i][1]= 1
    else:
         x_test[i][1]= 0
 #---------> complete test


x_target_1 = np.array(x)[:222,[1]]
x_target_2 = np.array(x)[554:690,[1]]

x_target = np.append(x_target_1,x_target_2) #------------> complete label test


#-----------------------------#
x_sex = np.array(x)[:,[4]]
for i in range (len(x_sex)):
    if  x_sex[i]== "male":
         x_sex[i]= 1
    else:
         x_sex[i]= 0
#-----------------------------#
x_em = np.array(x)[:,[11]]
for i in range (len(x_em)):
    if  x_em[i]== "Q":
         x_em[i]= 2
    elif x_em[i]=="S":
         x_em[i]= 0
    else:
        x_em[i]=1
#-----------------------------#
x_age = np.array(x)[:,[5]]
sum = 0
count = 0
for i in range (len(x_age)):
    if x_age[i] <99:
        sum += x_age[i]
        count+= 1
        
sum = sum/count

for i in range (len(x_age)):
    if x_age[i] < 99:
        count = count
    else:
        x_age[i] = int(sum)

'''
x_sex_1 = np.array(x)[:,[4]]
x_sex_2 = np.array(x)[:,[4]]
x_sex = np.append(x_sex_1,x_sex_2)

for i in range (len(x_sex)):
    if  x_sex[i]== "male":
         x_sex[i]= 1
    else:
         x_sex[i]= 0

x_sex = np.array(x_sex)

x_target_1 = np.array(x)[:50,[1]]
x_target_2 = np.array(x)[-50:,[1]]
x_target_1 += 5.5
x_target_2 += 5
x_target = np.append(x_target_1,x_target_2)


x_pclass_1 = np.array(x)[:50,[2]]
x_pclass_2 = np.array(x)[-50:,[2]]
x_pclass = np.append(x_pclass_1,x_pclass_2)

x_sib_1 = np.array(x)[:50,[6]]
x_sib_2 = np.array(x)[-50:,[6]]
x_sib = np.append(x_sib_1,x_sib_2)

x_par_1 = np.array(x)[:50,[7]]
x_par_2 = np.array(x)[-50:,[7]]
x_par = np.append(x_par_1,x_par_2)
'''



'''



'''
x_sex_1 = np.array(x_sex)[222:554]
x_sex_2 = np.array(x_sex)[690:]
x_sex = np.append(x_sex_1,x_sex_2)

x_label_1 = np.array(x)[222:554,[1]]
x_label_2 = np.array(x)[690:,[1]]
x_label = np.append(x_label_1,x_label_2)


x_pclass_1 = np.array(x)[222:554,[2]]
x_pclass_2 = np.array(x)[690:,[2]]
x_pclass = np.append(x_pclass_1,x_pclass_2)


x_sib = np.array(x)[:,[6]]
x_par = np.array(x)[:,[7]]

x_em = np.array(x_em)
x_age = np.array(x_age)

'''
plt.plot(x_target , 'r.' ,x_sib,'b.')
plt.show()
plt.plot(x_target , 'r.' , x_pclass ,'g.')
plt.show()
plt.plot(x_target , 'r.' , x_par ,'c.')
plt.show()
plt.plot(x_target , 'r.' , x_sex ,'k.')
plt.show()
plt.plot(x_target , 'r.' , x_em ,'m.')
plt.show()
plt.plot(x_target , 'r.' , x_age ,'y.')
plt.show()
plt.plot(x_target , 'r.' , x_pclass ,'g.',x_sib,'b.',x_par,'c.',x_sex,'k.', x_em ,'m.')
plt.show()
##sex pclass len 891 label [552 : 221 ] [339 : 136  (554+136)=>690]
'''
#x_test = np.reshape(x_test,(len(x_test),1))

'''
plt.plot(x_pclass,'r.',x_sex,'g.',C,'b.',B,'y.')
'''
acc = 0
a = 100
b = 1
sex_count = 0.1
pclass_count = 0.1
count = 0
'''
while(acc < 0.8):
    neigh = KNeighborsClassifier(n_neighbors=a,weights = 'uniform',algorithm = 'kd_tree')
    neigh.fit(BB, CC)
    neigh.predict(x_test)
    acc = neigh.score(x_test,x_target)
    print(acc , "====>" , a)
    a += 1
'''
cl = 10
sex = 5

x_sex = x_sex*sex
x_pclass += cl
x_train = x_sex+x_pclass 
x_train = np.reshape(x_train, (533,1))

x_test_1 = np.array(x_test)[:,[0]]
x_test_2 = np.array(x_test)[:,[1]]
x_test_1 = x_test_1 + cl
x_test_2 = x_test_2*sex

x_test_3 = x_test_1 * x_test_2
#x_test_1 = x_test_1*2

#x_test = np.add(x_test_2,x_test_1)
x_test = np.add(x_test_1,x_test_2)
x_test = np.array(x_test)
#print(x_test.shape)
#print('-----------------')
x_target = list(x_target)
x_label = list(x_label)

clf = svm.SVC(kernel='linear',C=100)
clf.fit(x_train,x_label)
pred = clf.predict(x_test)
acc = clf.score(x_test,x_target)
print(acc , "====>" , "======>" )
#plt.hist([x_train[:327],x_train[327:]])
'''
print(pred)
pred = clf.predict(final_sum)
print(pred)
df_2 = np.array(final)[:,[0]]
out_1 = df.DataFrame(df_2)
out_1.to_csv('final_1.csv')
out_2 = df.DataFrame(pred)
out_2.to_csv('final.csv')
'''