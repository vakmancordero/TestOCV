package testocv.train;

import processing.core.PApplet;
import testocv.psvm.SVM;
import testocv.psvm.SVMProblem;

public class Libsvm extends Classifier {
    
    private SVM classifier;
    
    public Libsvm(int numFeatures) {
        super(numFeatures);
    }
    
    float[] doubleToFloat(double[] input){
        
        float[] result = new float[input.length];
        
        for(int i = 0; i < input.length; i++){
            result[i] = (float)input[i];
        }
        
        return result;
    }
    
    @Override
    public void train() {
        
        float[][] trainingVectors = new float[trainingSamples.size()][trainingSamples.get(0).getFeatureVector().length];
        
        int[] labels = new int[trainingSamples.size()];
        
        for(int i = 0; i < trainingSamples.size(); i++) {
            trainingVectors[i] = doubleToFloat(trainingSamples.get(i).getFeatureVector());
            labels[i] = trainingSamples.get(i).getLabel();
        }
        
        classifier = new SVM(new PApplet());
        
        classifier.params.kernel_type = SVM.RBF_KERNEL;
        
        SVMProblem problem = new SVMProblem();
        
        System.out.println(numFeatures);
        
        problem.setNumFeatures(numFeatures);
        problem.setSampleData(labels, trainingVectors);
        
        classifier.train(problem);
    }
    
    public void save(String filename){
        classifier.saveModel(filename);
    }
    
    public void load(String filename){
        classifier.loadModel(filename, numFeatures);
    }
    
    @Override
    double predict(Sample sample) {
        return classifier.test(doubleToFloat(sample.getFeatureVector()));
    }
    
    public double predict(Sample sample, double[] confidence) {
        return classifier.test(doubleToFloat(sample.getFeatureVector()), confidence);
    }
}
