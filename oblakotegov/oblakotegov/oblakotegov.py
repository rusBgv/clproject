from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import os
from bs4 import BeautifulSoup   
import requests                 
import matplotlib.pyplot as plt
import os

class Content:
    def __init__(self, url, title, body):
        self.url   = url
        self.title = title
        self.body  = body


def get_page(url):
    req = requests.get(url)

    if req.status_code == 200:
        return BeautifulSoup(req.text, 'html.parser')
    return None 

def news_form_lenta(url):
    bs = get_page(url)
    if bs is None:
        return bs
    titleBs = bs.find("title")
    if titleBs:
        title = titleBs.text
    else: title = ' '
    lines = bs.find_all("p")
    body  = '\n'.join([line.text.strip() for line in lines])
    return Content(url, title, body)



content = news_form_lenta('https://www.wikipedia.org/')


if content is None:
    print("Ошибка!")
else:
    with open('fileOutput.txt', 'w') as f:

        print(content.body, file=f)
 


f = open('fileOutput.txt', 'r' )
 
text1=f.read()

stopwords = set(STOPWORDS)
stopwords.update(["на", "все", "эта", "не", "вчера", "уже"])

worldcloud=WordCloud(stopwords=stopwords, background_color="white").generate(text1)

plt.imshow(worldcloud, interpolation ="bilinear")
plt.axis("off")
plt.show()