#!/usr/bin/env python
# coding: utf-8

# In[9]:


def longestCommonPrefix(X):
    pref = X[0].lstrip()
    for S in X: 
        while (not S.lstrip().startswith(pref) and pref):
            pref = pref[:-1]
    return pref

