import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import text2emotion as t

import nltk
nltk.download('omw-1.4')

def emo(symbol):
	coms=0
	angry=0
	happy=0
	sad=0
	surprise=0
	fear=0
	txt=requests.get("https://finance.yahoo.com/quote/"+symbol+"/community?p="+symbol,headers={'User-Agent':''}).text
	#print("Loaded.",len(txt.split("\n")))
	soup = BeautifulSoup(txt, features="lxml")
	for p in soup.find_all('li'):
		if p.get("class")!=None:
			if p.get("class")[0]=="comment":
				comment=p.get_text(" ")
				comment=comment.split(' ', 2)[2]
				comment=comment.split("Reply")[0]
				if "More" in comment.split(" ")[-1]: comment=comment.rsplit(" ",1)[0]
				comment = re.sub(r'^https?:\/\/.*[\r\n]*', '', comment, flags=re.MULTILINE)
				if "@" in comment:
					comment=comment.rsplit("@",1)[0]
				#print(comment)
				e=t.get_emotion(comment)
				coms+=1
				angry+=e["Angry"]
				happy+=e["Happy"]
				sad+=e["Sad"]
				surprise+=e["Surprise"]
				fear+=e["Fear"]
	#DEBUG
	#print("Average emotions for",symbol)
	#print("Comments:",coms)
	#print("Angry:",angry/coms)
	#print("Happy:",happy/coms)
	#print("Sad:",sad/coms)
	#print("Surprise:",surprise/coms)
	#print("Fear:",fear/coms)
	if coms!=0:
		return angry/coms,happy/coms,sad/coms,surprise/coms,fear/coms
	else:
		return 0,0,0,0,0 ## no comment available for download

