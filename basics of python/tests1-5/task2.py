#!/usr/bin/env python
# coding: utf-8

# In[10]:


def is_palindrome(x):
    y = 0
    z = x
    while (z>0):
        y = y*10 + z%10
        z //= 10
    if x == y:
        return "YES"
    else:
        return "NO"