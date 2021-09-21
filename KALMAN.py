import matplotlib.pyplot as plt
import numpy as np
import csv

#THE EQUATIONS USED TO ESTIMATE THE COORDINATES AND VELOCITIES USING KALMAN FILTER
'''
X_predicted(k) = A*X(k-1) + B*U
P_predicted(k) = A*P(k-1)*A(T) 

K(k) = (P_predicted(k)*H(T)) / (H*P_predicted(k)*H(T) + R(k))

X(k) = X_predicted(k) + K(k)*(Y(k) - H*X_predicted(k))
P(k) = P_predicted(k) - P_predicted(k)*K(k)*H
'''

dt=1

#DEFINING THE H, A, INITIAL COVARIANCE, Q and R MATRICES
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
P = np.diag((372.99815102559614-368.18931566716645, 6.5966106132139-3.686804471625727e-06, 0.0990697000865573, 6.361599147637872))
R = np.array([[7, 0, 0, 0], [0, 7, 0, 0],[0, 0, 10, 0],[0, 0, 0, 10]],dtype=float)
Q=np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])

def kalman(A,X,P,Y,H,R):

    X=np.array(X,dtype=float)
    A=np.array(A,dtype=float)

    #CALCULATING X_PREDICTED USING X_predicted(k) = A*X(k-1) + B*U
    X_p = np.dot(A, X)

    #CALCULATING P_PREDICTED USING P_predicted(k) = A*P(k-1) 
    P_p = np.add(np.dot(A, np.dot(P, A.T)),Q)

    #CALCULATING THE KALMAN GAIN USING K(k) = (P_predicted(k)*H(T)) / (H*P_predicted(k)*H(T) + R(k))
    K=np.divide(np.dot(P_p,H.T),sum(np.dot(np.dot(H,P_p),H.T),R))

    #UPDATING THE X MATRIX USING X(k) = X_predicted(k) + K(k)*(Y(k) - H*X_predicted(k))
    X = np.add(X_p,np.dot(K,np.subtract(Y,np.dot(H,X_p))))

    #UPDATING THE P MATRIX USING P(k) = P_predicted(k) - P_predicted(k)*K(k)*H
    P = np.subtract(P_p,np.dot(P_p,np.dot(K,H)))

    return (X,P)


#READ THE FILE WITH MEASURED X,Y COORDINATES
with open('kalmann.txt', 'r') as file:

    k=0
    count=1

    reader = csv.reader(file)

    #ITERATE TO FIND THE UPDATED COORDINATES FOR EACH SET OF MEASUREMENTS
    for row in reader:

        if k==0:
            x_initial=float(row[0])
            y_initial=float(row[1])

            X=np.array([[x_initial],[y_initial],[0],[0]])
            k+=1

            x_prev=x_initial
            y_prev=y_initial
            X_prev=[x_initial,y_initial]

        else:
            x=float(row[0])
            y=float(row[1])
            v_x=float(row[2])
            v_y=float(row[3])
            
            Y=np.array([[x],[y],[v_x],[v_y]])
            
            (X,P)=kalman(A, X, P, Y, H, R)

            print("For state ",count)
            print("\nMeasured (x,y): (",row[0],",",row[1],")\n")
            print("Updated (x,y): (",X[0][0],",",X[1][0],")\n")
            print("Updated covariance Matrix: \n",P)
            print("\n........................................................................................................\n")
            count+=1

            #PLOT THE MEAUSURED VALUES
            plt.plot([x,x_prev],[y,y_prev],'r',linewidth=0.5)

            #PLOT THE UPDATED VALUES
            plt.plot([X[0],X_prev[0]],[X[1],X_prev[1]],'k',linewidth=0.5)

            x_prev=x
            y_prev=y
            X_prev=X

plt.show()