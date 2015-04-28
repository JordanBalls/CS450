package iris;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;

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
        String filename = "C:\\Users\\jorda_000\\Documents\\CS450\\iris.csv";
        
        DataSource source = new DataSource(filename);
        Instances data = source.getDataSet();
        
        // Set up data and filter
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(new Random());
        RemovePercentage remove = new RemovePercentage();
        
        // Set up training data
        remove.setPercentage(30);
        remove.setInputFormat(data);
        Instances train = Filter.useFilter(data, remove);
        
        // Set up the testing data
        remove.setInputFormat(data);
        remove.setInvertSelection(true);
        Instances test = Filter.useFilter(data, remove);
        
        // Apply classifier && run test
        HardCodedClassifier hardcode = new HardCodedClassifier();
        hardcode.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(hardcode, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
    
}
