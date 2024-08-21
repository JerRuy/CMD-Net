import numpy as np


#function  Combinations(com_m,com_n,last,cur,i)
#  if (i==com_n+1) then
#    table.insert(coms,cur:clone())
#   return
#  end
#  for idx = last+1, com_m do
#    cur[i] = idx
#    Combinations(com_m,com_n,idx,cur,i+1)
#  end
#end
#Combinations(com_m,com_n,0,torch.DoubleTensor(com_n),1)




pbatch = 8
com_m = pbatch*2
com_n = pbatch


temp = np.zeros([12870, 8])
#coms = []
coms = []  
cnt = 0

def combination(com_m, com_n, last, cur, i):
    
    #print(i)
    
    if (i == com_n ):
        #print(cur)
        global cnt
        temp[cnt] =  cur
        coms.append(temp[cnt])
        #coms.insert(len(coms),cur)
        #print(cur)    
        #print(coms[0])
        cnt = cnt + 1
        
        return 

      
    for idx in range(last+1 ,com_m +1):
    
        cur[i] = idx - 1
        combination(com_m, com_n, idx, cur, i+1)
    
    #print(cur)    
    
combination(com_m, com_n, 0, np.zeros([com_n]), 0)

   
   
   
   
        
if __name__ == '__main__':

    pbatch = 8
    com_m = pbatch*2
    com_n = pbatch
    #coms = np.zeros([12870,8])

    coms = []   
    cnt = 0 
    combination(com_m, com_n, 0, np.zeros([com_n]), 0)
    
    for i in range(12870):
    
        print(coms[i])
        
        
    print(len(coms))
    #print(coms)
    
    #a = [1, 2, 3, 4, 5, 6, 7, 8]
    #coms.append(a)
    #coms.append(a)
    #coms.append(a)
    #print(coms[1])
    
    