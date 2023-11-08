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


layer layers[model_length]; // layers array of type layer and size equal to lenght of model

void init_nn() {
	layers[0].size = model[0]; // layer 0 is only input layer, doesnt have weights/biases/nodes only size.
	for(int i = 1; i<model_length; i++){ // for layer in layers
		layers[i].size = model[i]; // size according to model given by user
		layers[i].nodes = (node *)malloc(sizeof(node)*layers[i].size); // initialise nodes array
		for (int j = 0; j<layers[i].size; j++){ //for node in nodes from layer
			layers[i].nodes[j].bias = (long double)0.0; //biases are 0
			layers[i].nodes[j].weights = (long double *)malloc(sizeof(long double)*layers[i-1].size); //weights array
			for (int k=0; k<layers[i-1].size; k++){ // for weights in node, # of weights=prev_nodes
				layers[i].nodes[j].weights[k] = (long double)init_weight(); //weights between 0 and 1
			}
		}
	}
}
void nn(long double input[], long double output[]) {
    for (int i=0; i<layers[0].size; i++) {
	    // input given to input layer
	    layers[0].nodes[i].output=input[i]; // input has to be of size of input layer. 
    }
    for (int k=1; k<model_length; k++){ // for k in layers 
	    for (int i=0; i<layers[k].size; i++) { // for nodes in layer
		    // node output = b + w1*x1 + w2*x2 * w3*x3 ...
		    long double* output = &layers[k].nodes[i].output; //output variable, at end will contain appropriate output of node
		    *output = layers[k].nodes[i].bias; // 1st add bias to output
						       //
		    // add weights * outputs_of_previous_node
		    for (int j=0; j<layers[k-1].size; j++){
			    *output += layers[k-1].nodes[j].output*layers[k].nodes[i].weights[j];
		    }
		    *output = sigmoid(*output); // sigmoid activation function applied
	    }
    }
    for (int i=0; i<model[model_length-1]; i++) output[i]=layers[model_length-1].nodes[i].output; // putting output in output array to be used later
}

void backprop(long double input[], long double obs[]) {
	int size = layers[model_length-1].size; //output size
	long double prd[size]; // to be used to store output of network
	nn(input, prd);

	// last layer is special because of cost deriv, will generalise this later
	for (int i = 0; i<size ;i++) {
		node* x = &layers[ml].nodes[i];
		x->sum = -2*(obs[i] - x->output) * x->output*(1 - x->output); // d_cost * d_activation
	}

	for(int j = model_length-1-1; j>0; j--) { // L to 1st layer
		for(int i=0; i<layers[j].size; i++) { // for i in nodes
			node* x = &layers[j].nodes[i]; // node variable
			for(int k=0; k<layers[j+1].size; k++) {
				node* x2 = &layers[j+1].nodes[k]; // node in next layer layer
				x->sum+= x2->sum*x2->weights[i]; // because the weight of the next layer is affecting our node, we are derivating with respect to that
			}
			x->sum = x->sum*x->output*(1-x->output); // derivative of sigmoid
			
		} 

	}

	long double rate = -0.05;

	//change weights and biases using calculated derivatives
	for(int j = model_length-1; j>0; j--){
		for(int i = 0; i<layers[j].size; i++){
			node* x = &layers[j].nodes[i];
			x->bias += x->sum*rate; 
			for(int k=0; k<layers[j-1].size;k++){
				x->weights[k] += x->sum*layers[j-1].nodes[k].output*rate;
			}
		}
	}
}

int main() {
	init_nn();
	long double arr[784];
	for(int i =0; i<784; i++) arr[i]=init_weight();
	long double output[model[model_length-1]];

	nn(arr, output);
	for (int i = 0; i<model[model_length-1]; i++) printf("%Lf\n", output[i]);

}
