#!/usr/bin/env python
# coding: utf-8

# In[110]:


class BankCard:  
    
    def __init__(self, s):
        self.total_sum = s
    
    @property
    def balance(self):
        if self.total_sum < 1:
            print("Not enough money to learn the balance.\n")
            raise ValueError
        else:
            self.total_sum -= 1
        return self.total_sum
    
    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            print("Not enough money to spent " + str(sum_spent) + " dollars.\n")
            raise ValueError
        else:
            self.total_sum -= sum_spent
            print("You spent " + str(sum_spent) + " dollars. " + str(self.total_sum) + " dollars are left.\n")

    def __repr__(self):  
        return "To learn the balance you should put the money on the card, spent some money or get the bank data. The last procedure is not free and costs 1 dollar."
        
    def put(self, sum_put):  
        self.total_sum += sum_put
        print("You put " + str(sum_put) + " dollars. " + str(self.total_sum) + " dollars are left.\n")
    

