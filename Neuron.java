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
        
        // Return decision
        if(sum > 0)
            return 1;
        else
            return 0;
        
    }
    
    // Getter for certain weight
    public double getWeight(int index)
    {
        return weights.get(index);
    }
    
}
