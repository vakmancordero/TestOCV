package testocv.training;

import java.net.URISyntaxException;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import org.opencv.core.Mat;
import org.opencv.core.Size;

import testocv.TestOCV;
import testocv.train.Sample;

/**
 *
 * @author VakSF
 */
public class FaceTest {
    
    private final TestOCV testOCV;
    private final int size;
    
    public FaceTest() throws URISyntaxException {
        this.size = 100;
        this.testOCV = new TestOCV(1728, this.size);
        this.testOCV.facesTraining();
    }
        
    @Test
    public void angry() {
        assertEquals("Enojado", this.check("/testocv/image/Training/enojado.jpg"));
    }
    
    @Test
    public void happy() {
        assertEquals("Felicidad", this.check("/testocv/image/Training/felicidad.jpg"));
    }
    
    @Test
    public void surprise() {
        assertEquals("Sorpresa", this.check("/testocv/image/Training/sorpresa.jpg"));
    }
    
    private String check(String path) {
        
        double[] confidences = new double[6];
        
        Mat scarface = this.testOCV.getScarface(path);
        Mat imageResize = this.testOCV.resize(scarface.clone(), new Size(size, size));
        Mat grayImage = this.testOCV.convertToGray(imageResize.clone());
        
        Sample sample = new Sample(testOCV.getHOGDescriptors(grayImage.clone()).toArray());
        
        int prediction = (int) testOCV.getSvmClassifier().predict(sample, confidences);
        
        return TestOCV.getKeysByValue(testOCV.getFaces(), prediction).iterator().next();
    }
    
}