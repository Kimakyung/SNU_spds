# 2020.Spring.Basic python & deep learning Study

This is study of **Basic python programming and deep learning** by Hakyung Kim.



## 0. Basic Python
### Prob 1. 소인수 분해
자연수를 입력받아 소인수 분해하는 함수
소인수 분해 결과는 (소수,지수)의 Tuple 로 반환

- 코드 예시
```python
prime_factorizer(30)
```
![enter image description here](http://drive.google.com/uc?export=view&id=1bGytEeaQPbkruXAhlhQkrdlZl2jF0P-E)
### Prob 2. 리스트 합치기
Nested list를 입력으로 받아 저장된 모든 정수를 반환
- 코드예시
```python
list1=[1,2,3]
list2=[[[1],3],[[4,2],5],6,[[7]]]
list_accumulator(list1)
list_accumulator(list2)
```
![enter image description here](http://drive.google.com/uc?export=view&id=19MH5_UQMfOj3EGlMW3YC3CFS58TIP_Ym)
### Prob 3. 주석 삭제
String을 입력받아 #로 표기된 주석을 삭제
- 코드예시
```python
code="""
x=1###x=2
######y=2
y=3###y=4
print(x+y)#
"""
comment_remover(code)
```
![enter image description here](http://drive.google.com/uc?export=view&id=1NEMXywgKj9lblGygxKZJLSCQ-nHppwSt)

## 1. Excel Using python
### Prob 1. 스프레드 시트 제작
Row 10개, Column 10개의 spread sheet 제작
각 셀은 변수를 저장 후 반환할 수 있다.

- 코드 예시
```python
sheet1=Spreadsheet()
sheet1.set_value("A3",5)
sheet1.set_value("C1","hello")
print(sheet1)
sheet1.set_value("C1","world")
sheet1.get_value("C1")
```
![enter image description here](http://drive.google.com/uc?export=view&id=16GBKFNEzgXkD6M058UPMhwbL-zvlzuz9)
### Prob 2. 함수가 사용가능한 스프레드 시트 제작
다른 셀 한개를 참조하는 lambda function을 특정셀에 입력할 수 있도록 기능을 제공하는 smart spread sheet 구현
- 코드 예시 
```python
sheet=SmartSpreadsheet()
sheet.set_value("A1",5)
sheet.set_function("A2",lambda x:x+1,"A1")
sheet.set_value("C4","Hello")
sheet.set_function("C5",lambda x:x+"!!","C4")
sheet.set_function("C6",lambda x:x+10,"A2")
print(sheet)
```
![enter image description here](http://drive.google.com/uc?export=view&id=1OPSV17g7Y6hKAZViB1N6LblF-w_j83c0)
## 2. Sorting algorithm
Bubble sort, Insertion sort, Merge sort, Quick sort, Radix sort를 구현하라

```python
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

```
![enter image description here](http://drive.google.com/uc?export=view&id=1ApaKBCIW0WjRBurNUuvXGTuupKhhAhFF)

## 3. Basic CNN
패션아이템 다중 분류 모델
Cross entropy -> MSE로 lost function을 변화시키는 모델 구현
**one-hot encoding 사용**
* 파이토치 

```python
# Define the loss
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Define the epochs
epochs = 5

trainingTime=0
train_losses, test_losses = [], []

for e in range(epochs): 
    Trainingstart = time.time()  # 시작 시간 저장
    running_loss = 0
    for images, labels in trainloader:
        # Flatten Fashion-MNIST images into a 784 long vector
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model.forward(images)
        y_one_hot=torch.zeros_like(output)
        y_one_hot.scatter_(1, labels.view(-1,1),1)
        loss = criterion(output, y_one_hot)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()        
    else:
        test_loss = 0
        accuracy = 0
    trainingTime+=time.time() - Trainingstart

    # Turn off gradients for validation, saves memory and computation
    with torch.no_grad():
      # Set the model to evaluation mode
        model.eval()
      
      # Validation pass
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.shape[0], -1)
            ps = model(images)
            y_one_hot=torch.zeros_like(ps)
            y_one_hot.scatter_(1, labels.view(-1,1),1)
            test_loss += criterion(ps,y_one_hot)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    
    model.train()

    print("Epoch: {}/{}..".format(e+1, epochs),
          "Training loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))


```
* Tensorflow
```python
from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import time
start=time.time()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['accuracy'])

def to_one_hot(labels,dimension=10):
    results=np.zeros((len(labels),dimension))
    for i, label in enumerate(labels):
        results[i,label]=1.
    return results
trainingstart=time.time()
model.fit(train_images, to_one_hot(train_labels), epochs=5, batch_size=32)
trainingend=time.time()
test_loss, test_acc = model.evaluate(test_images, to_one_hot(test_labels), verbose=2)
print(test_acc)
print("training_time:",trainingend-trainingstart)
print("Entire code execute time:",time.time()-start)
```
## 4. Music RNN
데이터 전처리 과정에 초점을 둔 Music RNN
- Sub 폴더에 포함된 모든 파일 리스트로 불러오기 (파일구조: clean-midi 폴더 안에 각 뮤지션 이름 별로 폴더가 형성되어 있고 그 안에 midi 파일이 ㅇ들어있음)
```python
import glob
import os
path="/home/jovyan/project/Music-Generation/clean_midi"
file_list=[]
for i in glob.glob(os.path.abspath(path+'/*/*.mid')):
    file_list.append(i)

```
- Error 데이터를 제외하고 Training data로 읽어오기

```python
try:
    trainset = NotesGenerationDataset('/home/jovyan/project/Music-Generation/clean_midi', longest_sequence_length=None)
except Exception as e:
    print(e)
```
- 그냥 읽으려고 할때 나타나는 error

![enter image description here](http://drive.google.com/uc?export=view&id=11ZcLNYHiJAFJQEJt8pWB8HdU0VBDRcgc)

(진행 중)
