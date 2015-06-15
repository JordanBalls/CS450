/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package iris;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * 
 */
public class Network extends Classifier{
   // Allows for checking of the classification of the node
    Instances trainingData; 
   // A network has multiple layers
   List<Layer> layers = new ArrayList<>();
   List<Integer> neuronsInEachLayer = new ArrayList<>();
   
   double learningRate = .001;
   // Bias input
   double bias = -1.0;
   int numIterations = 0;
   int inputCount = 0;
   
    @Override
    public void buildClassifier(Instances trainingSet) throws Exception 
    {
        trainingData = trainingSet;
        // Set the number of inputs to the network to the number of attributes
        // i.e., 4 for the IRIS set
        setInputCount(trainingSet.numAttributes() - 1);
        
        buildNetwork(inputCount, neuronsInEachLayer);
        
        List<Double> values = new ArrayList<>();
        //stores the values our output layer gave us after feed forward
        List<Double> finalValues = new ArrayList<>();
        
        for (int i = 0; i < trainingSet.numInstances(); i++)
        {
            // Set values of instance
            for (int j = 0; j < trainingSet.instance(i).numAttributes() - 1; j++)
            {
                values.add(trainingSet.instance(i).value(j));
            }
        //gets the new values of what we calculated for the classification
        //this is probably where we want to loop x amount of times
           for (int k = 0; k < numIterations; k++)
           {
              getOutputs(values);
              backPropogate(trainingSet.instance(i));
           } 
           values.clear(); // reset list
        }   
    }    
     
    
    @Override
    public double classifyInstance(Instance newInstance) throws Exception 
    {
       List<Double> values = new ArrayList<>();
       List<Double> finalValues = new ArrayList<>();
       
       for (int i = 0; i < newInstance.numAttributes() - 1; i++)
       {
           values.add(newInstance.value(i));
       }
       
       finalValues = getOutputs(values); 
       
       double maxIndex = 0;
       
       //sets maxIndex to the highest value we calculated, i.e., what we think
       //the classification of the flower is according to our calculations.
       for(int i = 0; i < finalValues.size(); i++)
       {
           if (finalValues.get(i) > finalValues.get((int)maxIndex))
           {
               maxIndex = i;
           }
       }
       
       return maxIndex;
    }
    
    
       
   public Network(List<Integer> neuronsPerLayer, int numIteration)
   {
       neuronsInEachLayer = neuronsPerLayer;
       numIterations = numIteration;
   }
   // Network Constructor
   public void buildNetwork(int inputCount, List<Integer> neuronsInLayer)
   {
       // Add 1 input to account for bias
       layers.add(new Layer(neuronsInLayer.get(0), inputCount + 1));
       
       for (int i = 1; i < neuronsInLayer.size(); i++)
       {
           layers.add(new Layer(neuronsInLayer.get(i), // neuronCount
                   neuronsInLayer.get(i-1) + 1));      // inputCount
       }
   }
   
   public void teach() throws Exception           
   {
       // Loop through all instances in training set
       for(int i = 0; i < trainingData.numInstances(); i++)
       {
           // If classification is wrong, we need to update weights
           if (trainingData.instance(i).value(4) != classifyInstance(trainingData.instance(i)))
           { 
               backPropogate(trainingData.instance(i));
           }
           // else do nothing
       }
   }
   
   // go through network and update weights accordingly
   public void backPropogate(Instance instance) throws Exception
   {
           // Loop through until we are happy with the classification
           // Start at the output layer and work back
           for (int i = layers.size() - 1; i >= 0; i--)
           {
               // Check for output layer
               if (i == layers.size() - 1)
               {
                   // Loop through neurons in output layer
                   for (int j = 0; j < neuronsInEachLayer.get(i); j++)
                   {
                       // Set variables for weight update equation
                       double activationValue = layers.get(i).getNeuron(j).outputValue;
                       double targetValue = 0;
                       double leftValue = 0;
                       // Set node index to classifications
                       if (instance.classValue() == j)
                       {
                           targetValue = 1;
                       }
                       
                       // Setting error value for node
                       layers.get(i).getNeuron(j).error = activationValue * (1 - activationValue) * (activationValue - targetValue);
                       double error = layers.get(i).getNeuron(j).error; // because im lazy
                       
                       // Loop and update weights
                       for (int k = 0; k < layers.get(i).getNeuron(j).weights.size(); k++)
                       {
                           leftValue = layers.get(i-1).getNeuron(j).outputValue; // get the K node in the layer to the left corresponding to the weight we're changing
                           layers.get(i).getNeuron(j).setNewWeight(k, (layers.get(i).getNeuron(j).getWeight(k) - (learningRate * error * leftValue))); // set new weight
                       }                       
                   }
               }
               // Else, it's a hidden layer
               else
               {
                   // Loop through neurons in hidden layer
                   for (int j = 0; j < neuronsInEachLayer.get(i); j++)
                   {
                       // Set variables for weight update
                       double activationValue = layers.get(i).getNeuron(j).outputValue;
                       double rightWeight = 0;
                       double rightError = 0;
                       double leftValue = 0;
                       
                       double sumWeightsWithError = 0;
                       // Calculate Sum of rightWeights * rightError
                       
                       for (int a = 0; a < layers.get(i+1).neurons.size(); a++) // loop through number of neurons to the right
                       {
                            sumWeightsWithError += (layers.get(i+1).neurons.get(a).getWeight(j) * layers.get(i+1).neurons.get(a).error);
                       }
                       
                       // Set error at the node
                       layers.get(i).getNeuron(j).error = (activationValue * (1 - activationValue) * sumWeightsWithError);
                       double error = layers.get(i).getNeuron(j).error; // because im lazy
                       
                       // Update weights loop
                       for(int k = 0; k < layers.get(i).getNeuron(j).weights.size(); k++)
                       {
//                           if (i == 0)
//                           {
//                               leftValue = layers.get(i).getNeuron(j).outputValue;
//                           }
//                           else
//                           {
                           leftValue = layers.get(i).getNeuron(j).outputValue;
                           //}
                           layers.get(i).getNeuron(j).setNewWeight(k, layers.get(i).getNeuron(j).getWeight(k) - (learningRate * error * leftValue));
                       }
                   }
               }
           }
           
           // End of Layers loop, update all weights in network
           // Loop through each layer
           for (int i = 0; i < layers.size(); i++)
           {
               // Loop through each neuron in this layer
               for (int j = 0; j < neuronsInEachLayer.get(i); j++)
               {
                   layers.get(i).getNeuron(j).updateWeights();
               }
           }
           
   }
   
   public List<Double> getOutputs(List<Double> inputs)
   {
       List<Double> outputs = new ArrayList<>(inputs);
       
       for (Layer layer : layers)
       {
           addBias(outputs);
           
           outputs = layer.outputResult(outputs);
       }
       
       return outputs;
   }
   
   // Adds the bias to the neuron
   public void addBias(List<Double> outputs)
   {
       outputs.add(bias);
   }
   
   // Sets the bias value
   public void setBias(Double bias)
   {
       this.bias = bias;
   }
   
   // Gets the layer at parameter index
   public Layer getLayer(int index)
   {
       return layers.get(index);
   }
    
   public int getInputCount()
   {
       return inputCount;
   }
   
   public void setInputCount(int count)
   {
       inputCount = count;
   }
}
