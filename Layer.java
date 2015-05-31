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
public class Layer {
    
    // Layers have multiple neurons
    List<Neuron> neurons = new ArrayList<>();
    
    // Constructor, provides an inputCount number to the
    // Neuron constructor as well
    public Layer(int neuronCount, int inputCount)
    {
        for (int i = 0; i < neuronCount; i++)
        {
            neurons.add(new Neuron(inputCount));
        }
    }
    
    // Produce classification
    public List<Double> outputResult(List<Double> inputs)
    {
        List<Double> outputs = new ArrayList<>();
        
        // Loop through and find the outputs of each neuron
        for (Neuron neuron : neurons)
        {
            outputs.add(neuron.outputResult(inputs));
        }
        
        return outputs;
    }
    
    // Neuron getter
    public Neuron getNeuron(int index)
    {
        return neurons.get(index);
    }
}
