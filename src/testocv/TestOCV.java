package testocv;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

/**
 *
 * @author VakSF
 */
public class TestOCV {
    
    private final CascadeClassifier classifier;
    
    public TestOCV() {
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        this.classifier = new CascadeClassifier(getPath(
                "/testocv/xml/haarcascade_frontalface_default.xml"));
        
        Mat scarface = this.getScarface("/testocv/image/image.jpg");
        
        if (scarface != null) {
            
            Mat grayImage = this.convertToGray(scarface.clone());
            
            MatOfFloat hogDescriptors = this.getHOGDescriptors(grayImage);
            
            float[] floats = hogDescriptors.toArray();
            
            System.out.println(floats.length);
            
        }
        
    }
    
    private MatOfFloat getHOGDescriptors(Mat image) {
        
        Size winsize = new Size(25, 50);
        Size blocksize = new Size(10, 20);
        Size blockStride = new Size(5, 10);
        Size cellsize = new Size(2, 2);
        int nbins = 9;
        
        HOGDescriptor hog = new HOGDescriptor(
                winsize, blocksize, blockStride, cellsize, nbins);
        
        Size winStride = new Size(8, 8);
        Size padding = new Size(0, 0);
        
        MatOfFloat descriptors = new MatOfFloat();
        MatOfPoint locations = new MatOfPoint();
        
        hog.compute(image.clone(), descriptors, winStride, padding, locations);
        
        return descriptors;
    }
    
    private Mat getScarface(String fileName) {
        
        Mat image = Imgcodecs.imread(getPath(fileName));
        MatOfRect facesRect = new MatOfRect();
        
        this.classifier.detectMultiScale(image, facesRect);
        
        List<Rect> faces = new ArrayList<>(Arrays.asList(facesRect.toArray()));
        
        return !faces.isEmpty() ? 
                new Mat(image, faces.iterator().next()) : null;
        
    }
    
    private Mat convertToGray(Mat mat) {
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY); return mat;
    }
    
    private String getPath(String url) {
        try {
            return new File(
                    getClass().getResource(url).toURI()
            ).getAbsolutePath();
        } catch (URISyntaxException ex) {}
        
        return null;
    }
    
    public static void main(String[] args) throws URISyntaxException, IOException {
        new TestOCV();
    }
    
}
