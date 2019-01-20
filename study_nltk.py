# -*- coding: UTF-8 -*-  
list = [1,2,3,4]
print(list)
print('First element:'+str(list[0]))
print('last element:'+str(list[-1]))
print('first three elements:'+str(list[0:2]))
print('last three elements:'+str(list[-3:]))
print('************************************')
#print(dir(list))
#print(help(list.index))
#print('************************************')
mystring='Monty Python ! And the holy Grail ! \n'
print(mystring.split())
print(mystring.strip())
print(mystring.rstrip())
print(mystring.lstrip())
print(mystring.upper())
print(mystring.replace('!',''))
print('************************************')
'''
import re
if re.search('Python',mystring):
    print('we found python')
else:
    print('no')
print(re.findall('!',mystring)) # ['!', '!']
print('**********  dictionary  **************************')
word_freq={}
for tok in mystring.split():
    if tok in word_freq:
        word_freq [tok] += 1
    else:
        word_freq [tok] = 1

print(word_freq)
'''
'''
print('**********  function  **************************')

import sys
def wordfreq(str):
    print(str)
    word_freq2 = {}
    for tok in str.split():
        if tok in word_freq2:
            word_freq2[tok] += 1
        else:
            word_freq2[tok] = 1
    print(word_freq2)

def main():
    str = 'This is my first python program'
    wordfreq(str)

if __name__ == '__main__':
    main()
'''

print('**********  nltk    **************************')
import urllib.request
url = 'http://www.baidu.com'
html = urllib.request.urlopen(url).read()
html = html.decode('utf-8')
print(len(html))
'''
print('*********   清理html标签   *************')
tokens = [tok for tok in html.split()]
print('Total no of tokens:'+str(len(tokens)))
print(tokens[0:100])
'''
import re
#tokens = re.split(r'\W+',html)  报错

#print(len(tokens))
#print(tokens[0:100])

import nltk
'''
from bs4 import BeautifulSoup #记得pip install beautifulsoup4
soup = BeautifulSoup(html,'html.parser')  #or  soup = BeautifulSoup(html,'html.parser')
clean = soup.get_text
#print(clean)
tokens = [tok for tok in clean().split()] #确保返回的是字符串，clean()
print(tokens[:100])
#用nltk进行词频的统计:
'''
#Freq_dist_nltk = nltk.FeatDict(tokens)
#print(Freq_dist_nltk[:100])
#for key, val in Freq_dist_nltk.items():
#    print(str(key)+':'+str(val))
'''
import operator
freq_dis={}
for tok in tokens:
    if tok in freq_dis:
        freq_dis[tok]+=1
    else:
        freq_dis[tok]=1

sorted_freq_dist=sorted(freq_dis.items(),key=operator.itemgetter(1),reverse=True)
print(sorted_freq_dist[:25])
#sorted_freq_dist.plot(50,cumulative=False)
'''
import nltk
from bs4 import BeautifulSoup
#clean = nltk.clean_html(html)
#tokens = [tok for tok in clean.split()]
soup = BeautifulSoup(html, "lxml")
clean = soup.get_text()
tokens = [tok for tok in clean.split()]
print(tokens[:100])
Freq_dist_nltk = nltk.FreqDist(tokens)
print(Freq_dist_nltk)
for k, v in Freq_dist_nltk.items():
    print(str(k) + ':' + str(v))

#Freq_dist_nltk.plot(50, cumulative=False)
## 停用词处理
stopwords = [word.strip().lower() for word in open("PATH/english.stop.txt")]
clean_tokens=[tok for tok in tokens if len(tok.lower()) > 1 and (tok.lower() not in stopwords)]
Freq_dist_nltk = nltk.FreqDist(clean_tokens)
Freq_dist_nltk.plot(50, cumulative = False)