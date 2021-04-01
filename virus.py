import numpy as np
import random
import matplotlib.pyplot as plt

def swap8(array):
    randomx1 = []
    randomy1 = []
    while len(randomx1) < 16:
        x = random.randrange(0,100)
        if x not in randomx1:
            randomx1.append(x)

    while len(randomx1) < 16:
        y = random.randrange(0,150)
        if y not in randomx1:
            randomx1.append(y)   

    for i in range(8):
        temp = array[randomx1[i],randomx1[i]] 
        array[randomx1[i],randomx1[i]] = array[randomx1[15-i],randomx1[15-i]]
        array[randomx1[15-i],randomx1[15-i]] = temp

def firstNeighbours(array,i,j):
    firstNeighs=[(i,j+1),(i,j-1),(i+1,j),(i-1,j),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]  
    allowed=[]
    for a in firstNeighs:
        if a[0]>=0 and a[0]<=99 and a[1]>=0 and a[1]<=149:
            allowed.append(a)       
    firstNeighs=[a for a in allowed]
    for t in firstNeighs:
          if array[t[0],t[1]]==0:
              array[t[0],t[1]]=random.choice([[1,0],[0.25,0.75]])[0]

def secondNeighbours(array,i,j):
    secondNeighs=[(i-2,j-2), (i-2,j-1), (i-2,j), (i-2,j+1), (i-2,j+2), (i-1,j-2), (i-1,j+2), (i,j-2), (i,j+2), (i+1,j-2), (i+1,j+2),(i+2,j-2), (i+2,j-1), (i+2,j), (i+2,j+1), (i+2,j+2)]
    allowed=[]
    for a in secondNeighs:
        if a[0]>=0 and a[0]<=99 and a[1]>=0 and a[1]<=149:
            allowed.append(a)       
    secondNeighs=[a for a in allowed]
    for t in secondNeighs:
          if array[t[0],t[1]]==0:
              array[t[0],t[1]]=random.choice([[1,0],[0.08,0.92]])[0]              
              




array = np.zeros((100,150),dtype=int)
array[50,75]=1
count=[1]
change=[0]
iterations=0
unaffected= True
affected=[]
while unaffected:
    unaffected= False
    swap8(array)
    for i in range(0,99):
        for j in range(0,149):
            if array[i,j]==1:
                affected.append((i,j))
            else:
                unaffected= True
    for people in set(affected):
        firstNeighbours(array,people[0],people[1])
        secondNeighbours(array,people[0],people[1]) 

    iterations+=1
    count.append(len(set(affected)))
    change.append(count[-1]-count[-2])       

x1 = []
for i in range(iterations+1):
    x1.append(i+1)   
y1 = count
plt.plot(x1,y1)
plt.xlabel('Number of iterations')
plt.ylabel('Number of ones in the matrix')   

x2 = x1
y2 = change
plt.plot(x2,y2)
plt.xlabel('Number of iterations')
plt.ylabel('Change ones in each iteration')
plt.show()
print("Maxima in Plot 2 : "+str(max(change)))










