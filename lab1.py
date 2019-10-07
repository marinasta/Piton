import numpy as np
import scipy as sio
import scipy.optimize as so
import matplotlib.pyplot as plt

###1###
A = np.array([[1, 2, 3], [-2, 3, 0], [5, 1, 4]])
B =np.array([[1,2,3],[2,4,6],[3,7,2]])
print(A)
print(B)

u = np.array([[-4], [1], [1]])
print(u)
v = np.array([[3], [2], [10]])
print(v)


###2###
C=np.random.random((100,100))
print(C)

w=np.random.random((100,1))
print(w)

###3###
print(A+B)
print(A.dot(B))
print(A.dot(A).dot(A))
print(A.dot(B).dot(A))
print(np.dot(v.T, A.T).dot(u+2*v))
print(u.dot(v))
print(C.dot(w))
print(w.T.dot(C))

###4###
matrix = np.array(range(0,20))
matrix = np.tile(matrix, (20,1))
matrix=matrix*matrix.T
print(matrix)

###5###
sio.savemat('File',{"A":A,"B":B})
file=sio.loadmat('File')["A"]
print(file)

###6###
sum=A[A>0].sum()+B[B>0].sum()
print(sum)

###7###
s1 = A.reshape((1,A.shape[0]*A.shape[1]))
s2 = s1[0,range(0, s1.shape[1], 2)]
print(s2)

###8###
print(np.linalg.inv(A))

try: print(np.linalg.inv(B))
except: print('не возможно найти, т.к. определить =0')

print(np.linalg.inv(C))

print(np.linalg.pinv(A))
print(np.linalg.pinv(B))
print(np.linalg.pinv(B))

###9###
a=np.array([[32,7,-6],[-5,-20,3],[0,1,-3]])
b=np.array([[12],[3],[7]])
result=np.linalg.solve(a,b)
print(result)

###10###
eigenvalues,eigenvectors=np.linalg.eig(A)
eigenvectors=np.matrix(eigenvectors).T
print("Собств. вектора = ",eigenvectors,'\n')
print("Собств. значения =",eigenvalues, '\n')

###11###
def f1(x):
    return 5*(x-2)**4-1/(x**2+8)
f1_min=so.minimize(f1,0.0,method='BFGS')
print(f1_min.x)

def f2(x):
   return 4*(x[0]-3*x[1])**2+7*x[0]**4
f2_min=so.minimize(f2,[0.0,1.0],method='BFGS')
print(f2_min.x)

###12###
def g1(x):
    return x**5-2*x**4+3*x-7
def g2(x):
    return x**5+2*x**4-3*x-7
x=np.linspace(-5,5,100)

y1=g1(x)
y2=g2(x)
plt.plot(x,y1)
plt.plot(x,y2)

plt.xlabel("x")
plt.ylabel("y1,y2")
plt.legend(['y1','y2'])
plt.title("title")

plt.show()

###13###
plt.subplot(2,1,1)
plt.plot(x,y1)
plt.xlabel("x")
plt.ylabel("y1")
plt.legend(['y1'])
plt.title("#1")

plt.subplot(2,1,2)
plt.plot(x,y2)
plt.xlabel("x")
plt.ylabel("y2")
plt.legend(['y2'])
plt.title("#2")

plt.show()

###14###
y1=so.brentq(g1,-5,5)
y2=so.brentq(g2,-5,5)

print(y1)
print(y2)





