package testocv.train;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.KNearest;

public class KNN extends Classifier {
    
    private KNearest classifier;
    
    public KNN(){
        super();
    }
    
    @Override
    public void train() {
        
        Mat trainingMat = new Mat(trainingSamples.size(), trainingSamples.get(0).getFeatureVector().length, CvType.CV_32FC1);
        Mat labelMat = new Mat(trainingSamples.size(), 1, CvType.CV_32FC1);
        
        // load samples into training and label mats.
        for (int i = 0; i < trainingSamples.size(); i++) {
            Sample trainingSample = trainingSamples.get(i);
            trainingMat.put(0, i, trainingSample.getFeatureVector());
            labelMat.put(i, 0, trainingSample.getLabel());
        }
        
        this.classifier = KNearest.create();
//        this.classifier.train(trainingMat, Ml.ROW_SAMPLE, );
    }
    
    @Override
    double predict(Sample sample) {
        
        Mat predictionTraits = new Mat(1, sample.getFeatureVector().length, CvType.CV_32FC1);
        predictionTraits.put(0, 0, sample.getFeatureVector());
        
        return classifier.findNearest(predictionTraits, 4, new Mat(), new Mat(), new Mat());
    }
}
