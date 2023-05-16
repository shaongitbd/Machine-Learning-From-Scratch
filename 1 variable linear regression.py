
import numpy as np

# We are inputting sample data in x and the value in ys
x = np.array([140.00,155.00,159.00,179.00], dtype=np.float64)

y  = np.array([60.00,62.00,67.00,70.00],dtype=np.float64 )
w = 0
b = 0
alpha = 1.0e-2
m = len(x)
iterations=50

gd  = []


# impletmenting our Linear Regression 

def model(x,w,b):
    return w*x+b


# cost function 

def cost_func(w,b):
   cost = 0
   for idx in range(m):
      cost +=  (model(x[idx],w,b)-y[idx])**2.00

   cost = cost/(2*m)
   return cost 
      

#  you can further optimize  it by combining both function and using one single loop

def update_w(w,b):
   value = 0

   for idx in range(m):
      value += (model(x[idx],w,b) -y[idx] )* x[idx] 

   return value
  
def update_b(w,b):
   value = 0
   for idx in range(m):
      value += model(x[idx],w,b) -y[idx] 
   return value 



def gradient_descent():
    
    global w
    global b
    
    for i in range(iterations):
   
      tempw = w
      w = tempw - alpha*((update_w(tempw,b))/m)
    
      tempb = b 
      b = tempb - alpha *((update_b(tempw,tempb))/m)
      
      cost  = cost_func(w,b)
      gd.append(cost)


         
         
         



    
gradient_descent()
    
print(gd[0:10],  gd[9990:99999])




