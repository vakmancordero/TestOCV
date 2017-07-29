package testocv;

import java.net.URISyntaxException;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import testocv.train.Sample;

/**
 *
 * @author VakSF
 */
public class PruebaDefinitiva {
    
    private final TestOCV testOCV;
    private final int size;
    
    public PruebaDefinitiva() throws URISyntaxException {
        this.size = 100;
        this.testOCV = new TestOCV(1728, this.size);
        this.testOCV.facesTraining();
    }
        
    public void angry() {
        
        if (this.check("/testocv/image/Training/enojado.jpg").equals("Enojado")) {
            System.out.println("Angry OK");
        }
        
    }
    
    public void happy() {
        
        if (this.check("/testocv/image/Training/felicidad.jpg").equals("Felicidad")) {
            System.out.println("Happy OK");
        }
        
    }
    
    public void surprise() {
        
        if (this.check("/testocv/image/Training/sorpresa.jpg").equals("Sorpresa")) {
            System.out.println("Surprise OK");
        }
        
    }
    
    public void fear() {
        
        if (this.check("/testocv/image/Training/miedo.jpg").equals("Miedo")) {
            System.out.println("Miedo OK");
        }
        
    }
    
    public void sad() {
        
        if (this.check("/testocv/image/Training/triste.jpg").equals("Triste")) {
            System.out.println("Triste OK");
        }
        
    }
    
    public void disgusto() {
        
        if (this.check("/testocv/image/Training/disgusto.jpg").equals("Disgusto")) {
            System.out.println("Disgusto OK");
        }
        
    }
    
    private int counter = 0;
    
    private String check(String path) {
        
        double[] confidences = new double[6];
        
        Mat scarface = this.testOCV.getScarface(path);
        Mat imageResize = this.testOCV.resize(scarface.clone(), new Size(size, size));
        Mat grayImage = this.testOCV.convertToGray(imageResize.clone());
        
        Imgcodecs.imwrite("asdasd" + (counter++) + ".jpg", grayImage);
        
        Sample sample = new Sample(testOCV.getHOGDescriptors(grayImage.clone()).toArray());
        
        int prediction = (int) testOCV.getSvmClassifier().predict(sample, confidences);
        
        return TestOCV.getKeysByValue(testOCV.getFaces(), prediction).iterator().next();
    }
    
    public static void main(String[] args) throws URISyntaxException {
        
        PruebaDefinitiva pruebaDefinitiva = new PruebaDefinitiva();
        
        pruebaDefinitiva.angry();
        pruebaDefinitiva.happy();
        pruebaDefinitiva.surprise();
        pruebaDefinitiva.fear();
        pruebaDefinitiva.sad();
        pruebaDefinitiva.disgusto();
        
    }
    
}
