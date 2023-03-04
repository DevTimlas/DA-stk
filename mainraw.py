import pygad
import pygad.nn
import pygad.gann
import numpy
import math
import data
import random
import time
from datetime import datetime
import pickle
import os

##class bcolors:
##    HEADER = '\033[95m'
##    OKBLUE = '\033[94m'
##    OKCYAN = '\033[96m'
##    OKGREEN = '\033[92m'
##    WARNING = '\033[93m'
##    FAIL = '\033[91m'
##    ENDC = '\033[0m'
##    BOLD = '\033[1m'
##    UNDERLINE = '\033[4m'

# colors not supported on windows command prompt
class bcolors:
    HEADER = ''
    OKBLUE = ''
    OKCYAN = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''
    BOLD = ''
    UNDERLINE = ''

def equalizer(inputs,outputs):
        pos_cnt=0
        pos=[]
        neg_cnt=0
        neg=[]
        for i in range(0,len(outputs)):
                if (outputs[i]==[0,1]).all():
                        neg_cnt+=1
                        neg.append(i)
                elif (outputs[i]==[1,0]).all():
                        pos_cnt+=1
                        pos.append(i)
        fff=[]
        fff2=[]
        for i in range(pos_cnt):
                n_idx=neg[i]
                p_idx=pos[i]
                fff.append(inputs[n_idx])
                fff.append(inputs[p_idx])
                fff2.append(outputs[n_idx])
                fff2.append(outputs[p_idx])
        return fff,fff2

def round_(lst):
        l=[]
        for i in lst:
                f=float(i)
                if not math.isnan(f):
                        #l.append(round(f))
                        if f>=0.5:
                                l.append(1)
                        else:
                                l.append(0)
                else:
                        l.append(0)
        if lst[0]>lst[1]:
                l=[1,0]
        else:
                l=[0,1]
        return l

def fullround(lst):
        rnd=[]
        for i in lst:
                rnd.append(round_(i))
        return rnd

def transform(data):
        f=[]
        for i in data:
                if i==1:f.append([0,1])
                else:f.append([1,0])
        return f

def moy(x,y):
        z=[]
        for i in range(0,len(x)):
                if x[i]==[1,0] and y[i]==[1,0]:
                        z.append([1,0])
                else:
                        z.append([0,1])
        return z


data_inputs = numpy.array([])
data_outputs = numpy.array([])

def fitness_func(solution, sol_idx):
    global GANN_instance, ga_instance, data_inputs, data_outputs
    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs)
    #print(fullround(predictions))
    #correct_predictions = numpy.where((fullround(predictions) == data_outputs).all(axis=1))[0].size

    solution_fitness = 0 #(correct_predictions/len(data_outputs))*100
    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance
    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Accuracy   = {fitness}".format(fitness=ga_instance.best_solution()[1]))



#introduced double model classification for better accuracy
#1..........................
GANN_instance = pygad.gann.GANN(num_solutions=50,
                                num_neurons_input=18,
                                num_neurons_hidden_layers=[50,45,20],
                                num_neurons_output=2,
                                hidden_activations=["relu","relu","relu"],
                                output_activation="softmax")
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
filename="gen2023-2-17"

ga_instance = pygad.load(filename=filename)

population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
#2..........................
GANN_instance2 = pygad.gann.GANN(num_solutions=50,
                                num_neurons_input=18,
                                num_neurons_hidden_layers=[50,45,20],
                                num_neurons_output=2,
                                hidden_activations=["relu","relu","relu"],
                                output_activation="softmax")
population_vectors2 = pygad.gann.population_as_vectors(population_networks=GANN_instance2.population_networks)
filename2="gen2023-2-21"

ga_instance2 = pygad.load(filename=filename2)

population_matrices2 = pygad.gann.population_as_matrices(population_networks=GANN_instance2.population_networks, population_vectors=ga_instance2.population)
GANN_instance2.update_population_trained_weights(population_trained_weights=population_matrices2)
#............................

def prediction(input_):
        prediction_ = pygad.nn.predict(last_layer=GANN_instance.population_networks[0], data_inputs=numpy.array([input_]), problem_type="classification")
        prediction_ = numpy.array(prediction_)
        return prediction_
def prediction2(input_):
        prediction_ = pygad.nn.predict(last_layer=GANN_instance2.population_networks[0], data_inputs=numpy.array([input_]), problem_type="classification")
        prediction_ = numpy.array(prediction_)
        return prediction_

def test():
	symbols=[]
	f=open("stocks2.txt","r").readlines()
	for i in f:
		symbols.append(i.replace("\n",""))
	fw=open("./last-results.txt","w+")
	fw.write("=== Date: "+str(datetime.now())+" ===\n")
	fw.write("=== Processed with genetic algorithm models "+filename+" and "+filename2+" ===\n")
	fw.flush()
	print(bcolors.OKGREEN+"\n\n=== gen2 started :) ==="+bcolors.ENDC)
	print(bcolors.OKCYAN+"=== Date: "+str(datetime.now())+" ==="+bcolors.ENDC)
	#print(bcolors.WARNING+"** heyy, isn't it a little late to check for stocks?!"+bcolors.ENDC)
	print(bcolors.OKCYAN+"=== Processed with models "+filename+" and "+filename2+" ==="+bcolors.ENDC)
	print(bcolors.OKCYAN+"=== neural nets trained using genetic algorithm ==="+bcolors.ENDC)

	cnt=0
	random.shuffle(symbols)
	print("Taking 200 symbols.")
	

	
	for i in symbols[:200]:
	
		t = ""
		dc = ""
		
		try:
			
			t = (f"Testing {i} {cnt}/200...")
			dat,svol=data.get(i,2)
			p=moy(transform(prediction(dat)),transform(prediction2(dat)))
			#print("Buy:",p[0][0],"Sell:",p[0][1])
			if p[0][0]>p[0][1] and p[0][0]>=0.55 and svol: #and p[0][0]<=0.9
				# print(bcolors.OKGREEN+"\n[*******] Buy",i,",Score=",p[0][0],",Sell score=",p[0][1],bcolors.ENDC,"\n")
				dc = (f'buy {i} score={p[0][0]} sell={p[0][1]}')
				fw.write("[*******] Buy "+i+" ,Score= "+str(p[0][0])+" ,Sell score= "+str(p[0][1])+"\n")
				fw.flush()

			time.sleep(1.2)
		
		except Exception as e:
			# print(e)
			time.sleep(4)
		
		cnt+=1

		yield (t, dc) 
		
	
	fw.close()
	#send_results()
	
from flask import Flask
import asyncio
import uvicorn


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

starlette_app = Starlette()

@starlette_app.route('/api/data')
async def get_data(request):
    data = {'value': 42}
    return JSONResponse(data)


async def app(scope, receive, send):
    await starlette_app(scope, receive, send)
    


@starlette_app.route('/sse')
async def sse(request):
    async def event_generator():
    	for a, b in test():
           	yield {'event': f'{a}', 'data': f'{b}'}
           	await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

	
if __name__ == "__main__":
	# uvicorn.run(app, host='0.0.0.0', port=4000)
	uvicorn.run("mainraw:app", host="0.0.0.0", port=os.getenv("PORT", default=5000), log_level="info")
	
	# print(dc)
# print("Check file \"last-results.txt\", list of stocks to buy for the short term!")
