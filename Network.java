/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package iris;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * 
 */
public class Network {
   // A network has multiple layers
   List<Layer> layers = new ArrayList<>();
   
   // Bias input
   double bias = 1.0;
   
   // Network Constructor
   public Network(int inputCount, List<Integer> neuronsInLayer)
   {
       // Add 1 input to account for bias
       layers.add(new Layer(neuronsInLayer.get(0), inputCount + 1));
       
       for (int i = 1; i < neuronsInLayer.size(); i++)
       {
           layers.add(new Layer(neuronsInLayer.get(i), // neuronCount
                   neuronsInLayer.get(i-1) + 1));      // inputCount
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
           
}
