import random
cnt=0
for i in range(0,10000):
    a=random.random()
    b=random.random()
    c=random.random()
    if a<=0.85 or (b<=0.95 and c<=0.9):
        cnt=cnt+1
ans=cnt/10000
print(ans)