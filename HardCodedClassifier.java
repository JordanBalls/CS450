package iris;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
/**
 *
 * @author Jordan Balls 4/27/2015
 */
public class HardCodedClassifier extends Classifier {
    @Override 
    public void buildClassifier(Instances instances) throws Exception {
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }
    
    
}
