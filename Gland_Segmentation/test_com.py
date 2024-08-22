import numpy as np

pbatch = 8
com_m = pbatch*2
com_n = pbatch


temp = np.zeros([12870, 8])
coms = []  
cnt = 0

def combination(com_m, com_n, last, cur, i):
    if (i == com_n ):
        #print(cur)
        global cnt
        temp[cnt] =  cur
        coms.append(temp[cnt])
        cnt = cnt + 1        
        return 
      
    for idx in range(last+1 ,com_m +1):    
        cur[i] = idx - 1
        combination(com_m, com_n, idx, cur, i+1)
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
    
    