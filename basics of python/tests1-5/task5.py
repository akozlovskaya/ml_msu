#!/usr/bin/env python
# coding: utf-8

# In[13]:


def is_pr(x):
    if not x%2:
        return x == 2
    y = 3
    while y*y <= x and x%y:
        y += 2
    return x < y*y

def primes():
    num = 2
    while True:
        if is_pr(num):
            yield num
        num += 1

