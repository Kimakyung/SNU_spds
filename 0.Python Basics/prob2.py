def list_accumulator(ls):
    res=0
    for i in ls:
        if type(i)==list:
            res+=list_accumulator(i)
        else:
            res+=i
    return res
