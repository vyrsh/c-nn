#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
using namespace std;

struct node
{
	long double *weights;
	long double bias;
	int *activation(long double);
	long double output;
	long double sum; 
};

struct layer
{
	int size;
	node *nodes;
};

// 784I - 10 - 10O
const int model[] = {784, 10, 10}; // specifies the size of each layer
const int model_length = sizeof(model)/sizeof(int);
const int ml = model_length-1; // abbr
			   
long double sigmoid(long double x)
{
    return 1 / (1 + exp(-x));
}

long double init_weight()
{
    return ((long double)rand())/((long double)RAND_MAX);
}


layer layers[model_length];
void init_nn() {
	layers[0].size = model[0]; // layer 0 is only input layer, doesnt have weights/biases/nodes only size.
	for(int i = 1; i<model_length; i++){ // for layer in layers
		layers[i].size = model[i]; // size according to model
		layers[i].nodes = (node *)malloc(sizeof(node)*layers[i].size); // initialise nod
		for (int j = 0; j<layers[i].size; j++){ //for node in nodes from layer
			layers[i].nodes[j].bias = (long double)0.0;
			layers[i].nodes[j].weights = (long double *)malloc(sizeof(long double)*layers[i-1].size);
			for (int k=0; k<layers[i-1].size; k++){ // for weights in node, # of weights=prev_nodes
				layers[i].nodes[j].weights[k] = (long double)init_weight();
			}
		}
	}
}


void nn(long double input[], long double output[]) {
    long double inp[model[0]];
    for (int i=0; i<model[0]; i++) inp[i]=input[i];
    long double sum;
    for (int k=1; k<model_length; k++){ // for k in layers 
	    for (int i=0; i<layers[k].size; i++) { // for nodes in layer
		    sum = layers[k].nodes[i].bias;
		    for (int j=0; j<layers[k-1].size; j++){
			    sum += inp[j] * layers[k].nodes[i].weights[j];
		    }
		    layers[k].nodes[i].output = sigmoid(sum); // output[i] = layers[k].nodes[i].activation(sum)
	    }
	    //for (int i = 0; i<model[model_length-1]; i++) printf("%Lf\n", out[i]);
	    long double inp[layers[k].size];
	    for (int i=0;i<layers[k].size; i++) inp[i]=layers[k].nodes[i].output;
	    
    }
    for (int i=0; i<model[model_length-1]; i++) output[i]=layers[model_length-1].nodes[i].output;
}

void backprop(long double input[], long double obs[]) {
	int size = model[model_length-1];
	long double prd[size];
	nn(input, prd);
	// last layer is special because of cost deriv
	for (int i = 0; i<size ;i++) {
		node x = layers[ml].nodes[i];
		x.sum = -2*(obs[i]-x.output)*x.output*(1-x.output);
	}

	for(int j = model_length-1-1; j>0; j--) { // L to 1st layer
		//iterate through nodes
		for(int i=0; i<model[j]; i++) {
			node x = layers[j].nodes[i];
			for(int k=0; k<model[j+1]; k++) {
				node x2 = layers[j+1].nodes[k]; 
				x.sum+= x2.sum*x2.weights[i];
			}
			x.sum = x.sum*x.output*(1-x.output);
		} 
			// record change
			// sum of error

	}
	long double rate = -0.05;
	//change weights and biases
	for(int j = model_length-1; j>0; j--){
		for(int i = 0; model[j];i++){
			node x = layers[j].nodes[i];
			x.bias += x.sum*rate;
			for(int k=0; k<model[j-1];k++){
				x.weights[k] += x.sum*layers[j-1].nodes[k].output;
			}
		}
	}
}


int main() {
	init_nn();
	long double arr[784];
	for(int i =0; i<784; i++) arr[i]=init_weight();
	long double output[model[model_length-1]];
	//for (int i = 0; i<model[0]; i++) printf("%Lf\n", arr[i]);
	//
	nn(arr, output);
	for (int i = 0; i<model[model_length-1]; i++) printf("%Lf\n", output[i]);

}
