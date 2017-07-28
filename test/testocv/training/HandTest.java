package testocv.training;

import java.net.URISyntaxException;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import testocv.TestOCV;
import testocv.train.Sample;

/**
 *
 * @author VakSF
 */
public class HandTest {
    
    private final TestOCV testOCV;
    private int size;
    
    public HandTest() throws URISyntaxException {
        this.size = 50;
        this.testOCV = new TestOCV(1728, this.size);
        this.testOCV.handTraining();
    }
    
    @Test
    public void A() {
        assertEquals("A", this.check("/testocv/data/test/A-uniform02.jpg"));
    }
    
    @Test
    public void B() {
        assertEquals("B", this.check("/testocv/data/test/B-uniform01.jpg"));
    }
    
    @Test
    public void C() {
        assertEquals("C", this.check("/testocv/data/test/C-uniform26.jpg"));
    }
    
    @Test
    public void V() {
        assertEquals("V", this.check("/testocv/data/test/V-uniform03.jpg"));
    }
    
    @Test
    public void five() {
        assertEquals("Five", this.check("/testocv/data/test/Five-uniform12.jpg"));
    }
    
    @Test
    public void point() {
        assertEquals("Point", this.check("/testocv/data/test/Point-uniform28.jpg"));
    }
    
    private String check(String path) {
        
        double[] confidences = new double[6];
        
        Mat image = Imgcodecs.imread(testOCV.getPath(path));
        Mat imageResize = this.testOCV.resize(image.clone(), new Size(size, size));
        Mat grayImage = this.testOCV.convertToGray(imageResize.clone());
        
        int prediction = (int) testOCV.getSvmClassifier().predict(
                new Sample(
                        testOCV.getHOGDescriptors(grayImage.clone()).toArray()
                ), confidences);
        
        String value = TestOCV.getKeysByValue(
                testOCV.getHands(), prediction).iterator().next();
        
        for (double confidence : confidences) 
            System.out.println(confidence);
        
        return value;
    }
    
}
