import emolib
import numpy as np
from sklearn.preprocessing import normalize
from yahoo_historical import Fetcher
from datetime import date
import time
import math
import statistics

# This library gets the input of a symbol for the new genetic trading algorithm

def diff(input):
	out=[]
	for i in range(1,len(input)):
		out.append((input[i]-input[i-1])/input[i-1])
	return out

def get(symbol,withlabels):
	#withlabels==2 : boolean [if last volume is greater than 100,000]
	today=date.today()
	from_=[]
	if today.month!=1:
		from_=[today.year,today.month-1,today.day]
	else:
		from_=[today.year-1,12,today.day]
	to_=[today.year,today.month,today.day]
	data=Fetcher(symbol,from_,to_).get_historical()
	prices=np.array(data["Close"][-7:])
	volume=np.array(data["Volume"][-7:])
	if withlabels==1:
		prices=np.array(data["Close"][-8:])
		volume=np.array(data["Volume"][-8:])
	pvar=normalize([diff(prices)])
	vvar=normalize([diff(volume)])
	r_=0.5
	labels=[0,0]
	if withlabels==1:
#		tanh=math.tanh(diff(prices)[-1]*100)
#		if diff(prices)[-1]>=0:
#			labels=[tanh,1-tanh]
#		else:
#			labels=[1+tanh,tanh]
		if diff(prices)[-1]>0:
			labels=[1,0]
		else:
			labels=[0,1]
	time.sleep(1.5)
	ang,hap,sad,sup,fea=emolib.emo(symbol)
	full=[]
	if withlabels==0 or withlabels==2:
		full.extend(pvar[0])
		full.extend(vvar[0])
	else:
		full.extend(pvar[0][:-1])
		full.extend(vvar[0][:-1])
	full.extend([r_])
	full.extend([ang,hap,sad,sup,fea])
	if withlabels==0 or withlabels==2:
		full=np.array(full)
		full=full.reshape(18)
	else:
		full.extend(labels)
		full=np.array(full)
		full=full.reshape(20)
	if withlabels==2:
		return full,statistics.mean(volume)>350000 #100000
	else:
		return full

def dl_all():
	symbols=[]
	f=open("stocks.txt","r").readlines()
	for i in f:
	        symbols.append(i.replace("\n",""))
	today=date.today()
	f=open("gendata"+str(today.year)+"-"+str(today.month)+"-"+str(today.day)+".csv","w+")
	count=0
	for i in symbols:
		try:
			print("Processing "+i+", "+str(count)+"/"+str(len(symbols))+" ...")
			tmp=get(i,0)
			#print(len(tmp))
			s=""
			for j in tmp:
				s+=(str(j)+",")
			s=s[:-1]
			s=i+","+s
			f.write(s+"\n")
			f.flush()
			count+=1
			time.sleep(1)
		except Exception as e:
			print("Network or parser error, 15-sec cooldown.")
			#print(str(e))
			count+=1
			time.sleep(10)
	f.close()

def dl_all_labels():
	symbols=[]
	f=open("stocks.txt","r").readlines()
	for i in f:
	        symbols.append(i.replace("\n","")) #symbols.append(i.split(",")[0])
	today=date.today()
	f=open("gendata"+str(today.year)+"-"+str(today.month)+"-"+str(today.day)+"labels.csv","w+")
	count=0
	for i in symbols:
		try:
			print("Processing "+i+", "+str(count)+"/"+str(len(symbols))+" ...")
			tmp=get(i,1)
			#print(len(tmp))
			s=""
			for j in tmp:
				s+=(str(j)+",")
			s=s[:-1]
			s=i+","+s
			f.write(s+"\n")
			f.flush()
			count+=1
			time.sleep(1)
		except Exception as e:
			print("Network or parser error, 15-sec cooldown.")
			count+=1
			time.sleep(10)
	f.close()

def load_labels(files):
	images=[]
	targets=[]
	for file in files:
		f=open(file,"r").readlines()
		for i in f:
			tmp=i.split(",")[1:]
			tmp=np.array(tmp)
			tmp=tmp.astype("float64")
			im=tmp[0:18]
			ta=tmp[18:20]
			images.append(im)
			targets.append(ta)
	return images, targets

#dl_all_labels()
