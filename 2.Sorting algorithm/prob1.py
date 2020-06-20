def bubble_sort(l):
    leng = len(l) - 1
    for i in range(leng):
        for j in range(leng-i):
            if l[j] > l[j+1]:
                l[j], l[j+1] = l[j+1], l[j]
        return l



def insertion_sort(l):
    j=1
    for j in range(j,len(l)):
        key=l[j]
        i=j-1
        while i>=0 and l[i]>key:
            l[i+1]=l[i]
            i=i-1
        l[i+1]=key
    return l



def merge_sort(l):
    if len(l) < 2:
        return l

    mid = len(l) // 2
    left = merge_sort(l[:mid])
    right = merge_sort(l[mid:])

    merged_l = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged_l.append(left[i])
            i += 1
        else:
            merged_l.append(right[j])
            j += 1
    merged_l += left[i:]
    merged_l += right[j:]
    return merged_l


def quick_sort(l):
    if len(l) <= 1:
        return l
    p = l[len(l) // 2]
    lesser_l, equal_l, greater_l = [], [], []
    for num in l:
        if num < p:
            lesser_l.append(num)
        elif num > p:
            greater_l.append(num)
        else:
            equal_l.append(num)
    return quick_sort(lesser_l) + equal_l + quick_sort(greater_l)


def radix_sort(l,d, base=10):
    def get_digit(number, d, base):
        return (number // base ** d) % base

    def counting_sort_with_digit(A, d, base):
        B = [-1] * len(A)
        k = base - 1
        C = [0] * (k + 1)
        for a in A:
            C[get_digit(a, d, base)] += 1
        for i in range(k):
            C[i + 1] += C[i]
        for j in reversed(range(len(A))):
            B[C[get_digit(A[j], d, base)] - 1] = A[j]
            C[get_digit(A[j], d, base)] -= 1
        return B

    digit = len(str(max(l)))
    for d in range(digit):
        l = counting_sort_with_digit(l, d, base)
