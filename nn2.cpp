#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
using namespace std;

struct node
{
    //long double value;
    int pnodes;
    long double *weights;
    long double bias;
};
// 784I - 10 - 10O

struct node layer1[10];
struct node layerO[10];

int model[] = {784,10,10};

long double sigmoid(long double x)
{
    return 1 / (1 + pow(2.71828182845904523536,-x));
}


void f_l(long double input[], struct node layer[], long double output[])
{
    for(int i = 0; i<784 ;i++) {
	   //printf("aa %d %Lf\n", i,input[i]);
    }
    long double sum;
    for (int i=0; i<10; i++)
    {
        sum = layer[i].bias;

        for (int j=0; j<layer[i].pnodes; j++)
        {

	        //printf("%Lf", input[j]);
		sum += input[j] * layer[i].weights[j];
        }

        output[i] = sigmoid(sum);
	//printf("%d %Lf\n", i,output[i]);
    }
}

void nn(long double input[], long double output[]) {
    long double layer1o[10];
    f_l(input, layer1, layer1o);
    f_l(layer1o, layerO, output);
}
long double init_weight()
{
    return ((long double)rand())/((long double)RAND_MAX*100);
}
void init_nbw(struct node layer[], int pn ) { //sets weights to random and biases to 0

    for (int j = 0; j<10; j++)
    {
        layer[j].pnodes = pn;
        layer[j].bias = (long double)0.0;
	layer[j].weights = (long double *)malloc(sizeof(long double)*pn);

        for (int i =0; i<layer[j].pnodes; i++)
        {
            layer[j].weights[i] = (long double)init_weight();
        }
    }

}


void init_nn() {
	init_nbw(layer1, 784);
	init_nbw(layerO, 10);
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



long double cost(long double a[], long double b[]) {
    long double cost = 0;
    for (int i=0; i<10; i++) {
        cost += pow((a[i] - b[i]) , 2);
    }
    return cost;
}

void print_image(long double pixels[]) { // print the 784 pixels
	for (int k = 0; k<784; k++){ 
		if (k%28==0) printf("\n"); // at 28 pixels print new line
		printf("%1.0Lf " , pixels[k]); 
	}
	printf("\n");
}


int mistakes = 0; //count number of mistakes, printed at end, in main();

// |||||||||||||||||||||||||||||||| BACKPROP ||||||||||||||||||||||||||||||||
void backprop(long double arr[], long double obs[]) { // input, observed
    // --Note generate prderved from input arr
    long double rate = -0.05;
    // cost = (prderved_i - predicted_i)^2 for i in output_nodes
    // slope = yf - yi / x2 - x1 (0.0001)
    long double output1[10];
    long double output[10];
    f_l(arr, layer1, output1);
    f_l(output1, layerO, output);
    printf("obs %d\n",max_index(output));
    if(max_index(output)!=max_index(obs)) {
	    printf("########\n");
	    mistakes+=1; }
    //for(int i=0; i<10; i++) printf("%d:%Lf ", i, output[i]);
    //long double yi = cost( output, obs );
    //printf("\nCost: %Lf\n", yi);
    // use diff cost / diff parameter to find slope
    // change =  and record new weight in array,
    // and will apply all new weights in end
    long double w1[10][784];
    long double w2[10][10];
    long double b1[10];
    long double b2[10];
    // nested for loops to iterate through every weight and bias.
    //layer1
    for (int n = 0; n < 10; n++) {
	    long double sum = 0;
	    for (int j=0; j<10; j++) {
		    sum+=-2*(obs[j]-output[j])*output[j]*(1-output[j])*layerO[j].weights[n];
	    }
	    //printf("%Lf %Lf\n",output[n]*(1-output[n]), output1[n]*(1-output1[n]));
	    long double slope = sum*output1[n]*(1-output1[n]);
	    b1[n] = slope*rate;
	    for (int i = 0; i < 784; i++) {
		    long double slope = sum*output1[n]*(1-output1[n])*arr[i];
		    w1[n][i] = slope*rate;
         }
    }
    //layer2
    for (int n = 0; n < 10; n++){
	    long double prd = output[n];
	    long double ct = obs[n] - prd;
	    long double slope = -2*ct*prd*(1-prd);
	    
	    b2[n] = slope*rate;
	    for (int i = 0; i < 10; i++) {
		    long double slope = -2*ct*prd*(1-prd)*output1[i];
		    w2[n][i] = slope*rate;
         }
    }

    // put stored values back into NN
    for (int n=0; n<10; n++) {
        layer1[n].bias += b1[n];
        layerO[n].bias += b2[n];
        for (int i=0; i<10; i++) {
            layerO[n].weights[i] += w2[n][i];
	}
	for (int i=0; i<784; i++) {

            layer1[n].weights[i] += w1[n][i];

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
			print_image(input); // comment out to decrease IO lag
			printf("%d label:%d ", i/784, labels[(i/784) - 1]);
			backprop(input, obs);
		}

	}
}


int main() {

	// load image and labels in array
	ifstream file1("train-images-idx3-ubyte", ios::in | ios::binary);
	static unsigned char pixels[47040000];
	ifstream file2("train-labels-idx1-ubyte", ios::in | ios::binary);
	unsigned char labels[60000];
	file1.read((char*)&pixels[0], 47040000); // pixels in array. each image is 784 pixels
	file2.read((char*)&labels[0], 60000); // labels

	init_nn(); 
	for (int i=0; i<2; i++) train(pixels, labels); // number of times train is run. set to 15 for best result.
	printf("%d", mistakes); //prints number of mistakes made. // least mistakes > 3924/60,000 > 93% accuracy 


}


