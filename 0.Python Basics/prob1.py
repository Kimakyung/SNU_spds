def  prime_def(number):
    if number!=1:
        for n in range (2,number):
            if number % n==0:
                return False
            else:
                return True
    return True
          
    
def prime_number_list(number):
    interger_list=(x for x in range (2,number+1))
    prime_numbers=[]
    for num in interger_list:
        if prime_def(num):
            prime_numbers.append(num)
    return prime_numbers

def prime_factorizer(number):
    res={}
    prime_numbers=prime_number_list(number)
    for p in prime_numbers:
        exp=0
        while number%p==0:
            exp+=1
            number=number//p
        res[p]=exp
        if number==1:
            break
    return(list(res.items()))


