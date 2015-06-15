package iris;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author Jordan Balls 4/27/2015
 * https://weka.wikispaces.com/Use+WEKA+in+your+Java+code
 * The above link was used as a source for a majority of this code
 */
public class Iris {

    /**
     * Reads in a csv file, creates a training and testing data
     * set from it, and runs it with the hard coded classifier
     */
    public static void main(String[] args) throws Exception {
        String file = "C:\\Users\\jorda_000\\Documents\\CS450\\iris.csv";
        
        double number = .7;
 
        DataSource source = new DataSource(file); // sets the source to the data file
        Instances data = source.getDataSet(); // gets the source and stores it into data

        data.setClassIndex(data.numAttributes()-1); // sets the index to 4 instead of 5 
                                                       // because 
        data.randomize(new Random()); // randomizes the data set
        
        // the maximum number of data points we want, and we want 70% of them thus we multiple
        // by 'number' which is .7
        int trainIndex = (int) Math.round(data.numInstances() * number); 
        
        // we want 30% of the data to test so we take all the data and subtract the trainIndex because
        // its 70% of the data so therefore we will have 100% - 70% = 30%
        int testIndex = data.numInstances() - trainIndex;
        
        // this is another instance of the dataset but its 70% of the data we use the number generated
        // for the trainIndex as the maximum number of data points we want from the dataset
        // it starts at 0 and goes until the max number
        Instances trainSet = new Instances(data, 0, trainIndex);
        Instances testSet = new Instances(data, trainIndex, testIndex);
        
        //standardizes the trainSet
        Standardize standardizedData = new Standardize();
        standardizedData.setInputFormat(trainSet);
        
        // Discretize the trainSet
        //Discretize standardizedData = new Discretize();
        //standardizedData.setBins(3);
        //standardizedData.setInputFormat(trainSet);
        
        //makes a new instance of data to test on the classifier
        Instances newTestSet = Filter.useFilter(testSet, standardizedData);
        Instances newTrainSet = Filter.useFilter(trainSet, standardizedData);
 
        //HardCodedClassifier hc = new HardCodedClassifier();
        //hc.buildClassifier(data);
        
        //KNearestNeighbor knn = new KNearestNeighbor();
        //knn.buildClassifier(newTrainSet);
        
        // Number of neurons in each layer
        List<Integer> neuronsPerLayer = new ArrayList<>();
        neuronsPerLayer.add(newTrainSet.numAttributes() - 1);   // 4 inputs
        neuronsPerLayer.add(newTrainSet.numClasses());          // 3 classifications
        int numIterations = 20000;
        
        Network neuralNetwork = new Network(neuronsPerLayer, numIterations);
        neuralNetwork.buildClassifier(newTrainSet);
        
        // evaluates the trainset
        Evaluation eval = new Evaluation(newTrainSet);
        
                
        // formates the evaluation based on the setting in the evaluateModel function format
        eval.evaluateModel(neuralNetwork, newTestSet);
        System.out.println(eval.toSummaryString("\n RESULTS \n", true));
        
    }
    
}
        
