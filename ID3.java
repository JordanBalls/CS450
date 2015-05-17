/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package iris;

import weka.classifiers.Classifier;
import weka.core.Instances;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

/**
 *
 * @author jorda_000
 */
public class ID3 extends Classifier {
    int usedAttributes = 0;
    int numInBin;
    Attribute highestInfoGain;
    Attribute classAttribute;
    double [] maxDistribution;
    double classValue;
    ID3 [] children;
    // Steps:
    // 1. Calculate the entropy of every attribute using the data set
    // 2. Split the set into subsets using the attribute for which entropy
    // is minimum(or, equivalently, information gain is maximum)
    // 3. Make a decision tree node containing that attribute
    // 4. Recur on subsets using remaining attributes
    public void makeLikeAWhat(Instances instances)
    {
        // Create storage for different info gains
        double [] infoGains = new double[instances.numAttributes()];
        // Enumerate through attributes to find the best gain
        Enumeration attributeEnum = instances.enumerateAttributes();
        while (attributeEnum.hasMoreElements())
        {
            // Loop through attributes, adding gain to infoGains array
            Attribute att = (Attribute) attributeEnum.nextElement();
            infoGains[att.index()] = infoGain(instances, att);
        }
        // Use maxIndex to find the highest info gain in the array
        highestInfoGain = instances.attribute(Utils.maxIndex(infoGains));
        
        // Make a leaf if there is no more info to gain
        // Otherwise, create children
        // Check if there is no more info to gain
        if (Utils.eq(infoGains[highestInfoGain.index()], 0))
        {
            highestInfoGain = null;
            // Instantiate maxDistribution
            maxDistribution = new double[instances.numClasses()];
            // Set up enumerator for instances
            Enumeration instanceEnum = instances.enumerateInstances();
            // Tally classes
            while (instanceEnum.hasMoreElements())
            {
                Instance instance = (Instance) instanceEnum.nextElement();
                maxDistribution[(int) instance.classValue()]++;
            }
            // Normalize data for easier manipulation
            Utils.normalize(maxDistribution);
            // Get the max index of the distrubtion
            classValue = Utils.maxIndex(maxDistribution);
            // Save class attribute
            classAttribute = instances.classAttribute();
        }
        // Create children
        else
        {
            // Split best attribute into bins
            Instances [] bins = makeBins(instances, highestInfoGain);
            // Create nodes
            children = new ID3[highestInfoGain.numValues()];
            for (int i = 0; i < highestInfoGain.numValues(); i++)
            {
                children[i] = new ID3();
                children[i].makeLikeAWhat(bins[i]);
            }
        }
    }
    
    public double calculateEntropy(Instances instances)
    {        
        // Array to hold counts for each class
        double [] numInEachClass = new double[instances.numClasses()];
        
        // Loop through every instance in one bin
        for (int i = 0; i < instances.numInstances(); i ++)
        {
            // Increment the count for the class that the instance belongs to
            numInEachClass[(int)instances.instance(i).classValue()]++;
        }
        // Instantiate the entropy value
        double entropy = 0;
        
        // Loop through number of classes to sum log operations
        for (int i = 0; i < instances.numClasses(); i++)
        {
            // Handle missing data
            if (numInEachClass[i] > 0)
            {
                // Logarithm algorithm for entropy
                entropy -= (numInEachClass[i]/instances.numInstances()) * Utils.log2(numInEachClass[i]/instances.numInstances());
            }
        }
        return entropy;
    }
        
    public double infoGain(Instances instances, Attribute att)
    {
        // Calculate total entropy
        double infoGain = calculateEntropy(instances);
        // Create bins
        Instances[] bins = makeBins(instances, att);
        // Loop through number of bins in attribute
        for(int i = 0; i < att.numValues(); i++)
        {
            // Applies weight to entropy value
            infoGain -= ((double) bins[i].numInstances() / (double) instances.numInstances()) *
                    calculateEntropy(bins[i]);                    
        }
        return infoGain;
    }
    
    private Instances[] makeBins(Instances instances, Attribute att)
    {
        // Create array of bins based on numValues in Attribute parameter
        Instances[] bins = new Instances[att.numValues()];
        
        for (int i = 0; i < att.numValues(); i++)
        {
            bins[i] = new Instances(instances, instances.numInstances());
        }
        
        // Create pointer to first instance
        Enumeration instanceEnum = instances.enumerateInstances();
        
        while(instanceEnum.hasMoreElements())
        {
            // Create new instance from the one pointer is pointing at
            Instance oneInstance = (Instance) instanceEnum.nextElement();
            // Add instance to the proper bin
            bins[(int)oneInstance.value(att)].add(oneInstance);
        }
    
        // Compactify
        for (int i = 0; i < bins.length; i++)
        {
            bins[i].compactify();
        }
        return bins;
    }
    
    
    @Override
    public void buildClassifier(Instances instance) throws Exception 
    {
        instance = new Instances(instance);
        
        // Removes instances that have missing data
        instance.deleteWithMissingClass();
        
        makeLikeAWhat(instance);
    }

    public double classifyInstance(Instance instance) throws Exception
    {
        // Assign class value
        if (highestInfoGain == null)
        {
            return classValue;
        }
        // Else, keep going
        else
        {
            return children[(int) instance.value(highestInfoGain)].classifyInstance(instance);
        }
    }
    
}
