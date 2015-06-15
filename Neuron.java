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
 * @author jorda_000
 */
public class Neuron {
    
    List<Double> weights = new ArrayList<>();
    List<Double> newWeights = weights;
    double error = 0;
    double outputValue = 0;
    
    // Constructor
    public Neuron(int inputCount) // pass in number of nodes
    {
        for (int i = 0; i < inputCount; i++)
        {
            weights.add(Math.random() * 2.0 - 1.0);
        }
    }
    
    // Determine if inputs classify under this node
    public double outputResult(List<Double> inputs)
    {
        double sum = 0;
        // Add the various (weight * input value)
        for (int i = 0; i < weights.size(); i++)
        {
            sum += weights.get(i) * inputs.get(i);
        }
        
        outputValue = (1 / (1 + Math.pow(Math.E, sum * -1)));
        // Return activation value
        return outputValue;
        
    }
    
    // Getter for certain weight
    public double getWeight(int index)
    {
        return weights.get(index);
    }
    
    public void setNewWeight(int index, double value)
    {
        newWeights.set(index, value);
    }
    
    public void updateWeights()
    {
        weights = newWeights;
    }
}
