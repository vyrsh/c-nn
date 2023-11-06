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
    return 0.00271828; //((long double)rand())/((long double)RAND_MAX*100);
}

int mistakes = 0;
layer layers[model_length];
void init_nn() {
	layers[0].size = model[0]; // layer 0 is only input layer, doesnt have weights/biases/nodes only size.
	for(int i = 0; i<model_length; i++){ // for layer in layers
		layers[i].size = model[i]; // size according to model
		layers[i].nodes = (node *)malloc(sizeof(node)*layers[i].size); // initialise nod
		for (int j = 0; j<layers[i].size && i!=0; j++){ //for node in nodes from layer
			layers[i].nodes[j].bias = (long double)0.0;
			layers[i].nodes[j].weights = (long double *)malloc(sizeof(long double)*layers[i-1].size);
			for (int k=0; k<layers[i-1].size && i!=0; k++){ // for weights in node, # of weights=prev_nodes
				layers[i].nodes[j].weights[k] = (long double)init_weight();
			}
		}
	}
}
void print_image(long double pixels[]) { // print the 784 pixels
	for (int k = 0; k<784; k++){ 
		if (k%28==0) printf("\n"); // at 28 pixels print new line
		printf("%1.0Lf " , pixels[k]); 
	}
	printf("\n");
}

void nn(long double input[], long double output[]) {
    //long double inp[model[0]];
    for (int i=0; i<model[0]; i++) {
	    //inp[i]=input[i];
	    layers[0].nodes[i].output=input[i];
    }
    //printf("NN");
    //print_image(inp);
    //long double sum;
    for (int k=1; k<model_length; k++){ // for k in layers 
	    for (int i=0; i<layers[k].size; i++) { // for nodes in layer
		    long double* output = &layers[k].nodes[i].output;
		    *output = layers[k].nodes[i].bias;
		    for (int j=0; j<layers[k-1].size; j++){
			    *output += layers[k-1].nodes[j].output*layers[k].nodes[i].weights[j];
			    //printf("%Lf ",inp[j] * layers[k].nodes[i].weights[j]);
		    }
		    *output = sigmoid(*output); // output[i] = layers[k].nodes[i].activation(sum)
	    }
	    //long double inp[layers[k].size];
	    //for (int i=0;i<layers[k].size; i++) inp[i]=layers[k].nodes[i].output;
	    
    }
    for (int i=0; i<model[model_length-1]; i++) output[i]=layers[model_length-1].nodes[i].output;
}

int max_index(long double arr[])
{
    int max = 0;
    for (int i=0; i<10; i++)
    {
        if (arr[i]>arr[max])
        {
            max = i;
        }
    }
    return max;
}

void backprop(long double input[], long double obs[]) {
	int size = model[model_length-1];
	long double prd[size];
	nn(input, prd);
	printf("obs %d\n",max_index(prd));

	//for (int i=0; i<size; i++) printf("%Lf",prd[i]); 
	// last layer is special because of cost deriv
	for (int i = 0; i<size ;i++) {
		node* x = &layers[ml].nodes[i];
		x->sum = -2*(obs[i] - x->output)*x->output*(1 - x->output);
		//printf("%Lf %Lf\n", x.sum, -2*(obs[i]-x.output)*x.output*(1-x.output));
	}

	for(int j = model_length-1-1; j>0; j--) { // L to 1st layer
		//iterate through nodes
		for(int i=0; i<model[j]; i++) {
			node* x = &layers[j].nodes[i];
			for(int k=0; k<model[j+1]; k++) {
				node* x2 = &layers[j+1].nodes[k]; 
				x->sum+= x2->sum*x2->weights[i];
			}
			x->sum = x->sum*x->output*(1-x->output);
			
		} 

	}
	long double rate = -0.05;
	//change weights and biases
	for(int j = model_length-1; j>0; j--){
		for(int i = 0; i<model[j]; i++){
			node* x = &layers[j].nodes[i];
			x->bias += x->sum*rate;
			//printf("s:%Lf\n", x.sum);
			for(int k=0; k<model[j-1];k++){
				x->weights[k] += x->sum*layers[j-1].nodes[k].output*rate;
				//printf("%Lf\n", x.weights[k]);
			}
		}
	}
}




void train(unsigned char pixels[], unsigned char labels[]) {
	long double input[784];
	mistakes=0;
	for (int i = 1; i<47040000; i++){ // pass 784 pixels into backprop
		input[i%784] = (long double)pixels[i]/255; //divide by 255 to put values below 1
		if (i%784==0) { // train
			long double obs[] = {0,0,0,0,0,0,0,0,0,0};
			obs[labels[(i/784) - 1]] = 1; // desired index/number is set to 1
			//print_image(input); // comment out to decrease IO lag
			printf("%d label:%d ", i/784, labels[(i/784) - 1]);
			backprop(input, obs);
			//return;
		}

	}
}


int main() {
	//init_nn();
	//long double arr[784];
	//for(int i =0; i<784; i++) arr[i]=init_weight();
	//long double output[model[model_length-1]];
	//for (int i = 0; i<model[0]; i++) printf("%Lf\n", arr[i]);
	//nn(arr, output);
	//for (int i = 0; i<model[model_length-1]; i++) printf("%Lf\n", output[i]);
	//printf("bbb");
	
	// load image and labels in array
	ifstream file1("train-images-idx3-ubyte", ios::in | ios::binary);
	static unsigned char pixels[47040000];
	ifstream file2("train-labels-idx1-ubyte", ios::in | ios::binary);
	unsigned char labels[60000];
	file1.read((char*)&pixels[0], 47040000); // pixels in array. each image is 784 pixels
	file2.read((char*)&labels[0], 60000); // labels
	init_nn();
	for (int i=0; i<1; i++) train(pixels, labels); // number of times train is run. set to 15 for best result.
	//printf("%d", mistakes); //prints number of mistakes made. // least mistakes > 3924/60,000 > 93% accuracy

}


