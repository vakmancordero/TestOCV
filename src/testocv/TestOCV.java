package testocv;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

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

import testocv.train.Libsvm;
import testocv.train.Sample;

/**
 *
 * @author VakSF
 */
public class TestOCV {
    
    private final CascadeClassifier classifier;
    
    private final Libsvm svmClassifier = new Libsvm(1728);
    
    private final Map<String, Integer> classNames = initClassNames();
    
    private Map<String, Integer> initClassNames() {
        
        Map<String, Integer> map = new HashMap<>();
        
        map.put("Enojado", 0);
        map.put("Sorpresa", 1);
        map.put("Miedo", 2);
        map.put("Felicidad", 3);
        map.put("Triste", 4);
        map.put("Disgusto", 5);
        
        return map;
    }

    public Map<String, Integer> getClassNames() {
        return classNames;
    }
    
    public TestOCV() {
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        this.classifier = new CascadeClassifier(getPath(
                "/testocv/xml/haarcascade_frontalface_default.xml"));
        
    }

    public Libsvm getSvmClassifier() {
        return svmClassifier;
    }
    
    public void generateVectors() throws URISyntaxException, IOException {
        
        StringBuilder builder = new StringBuilder();
        
        for (File file : this.getFile("/testocv/image").listFiles()) {
            
            String fileName = file.getName();
            
            Mat scarface = this.getScarface("/testocv/image/" + fileName);
            
            if (scarface != null) {
                
                Mat imageResize = this.resize(scarface.clone(), new Size(32, 32));
                
                Mat grayImage = this.convertToGray(imageResize.clone());
                
                MatOfFloat hogDescriptors = this.getHOGDescriptors(grayImage);
                
                builder
                        .append("\"")
                        .append(fileName)
                        .append("\" => ")
                        .append(hogDescriptors.toList().toString())
                        .append("\n");
                
            }
            
        }
        
        this.writeFile(builder, "output.txt");
        
    }
    
    public void training() throws URISyntaxException {
        
        for (File folder : this.getFile("/testocv/image").listFiles()) {
            
            if (folder.isDirectory()) {
                
                String folderName = folder.getName();
                
                int classNumber = this.classNames.get(folderName);
                
                System.out.println(folderName + " : " + classNumber);
                
                for (File file : folder.listFiles()) {
                    
                    Mat scarface = this.getScarface(
                            "/testocv/image/" + folderName + "/" + file.getName());
                    
                    String extension = file.getName().split("\\.(?=[^\\.]+$)")[1];
                    
                    if (extension.equalsIgnoreCase("jpg") || 
                            extension.equalsIgnoreCase("jpg") ||
                                extension.equalsIgnoreCase("jpeg")) {

                        if (scarface != null) {
                            
                            System.out.println(file.getName() + " : " + scarface);

                            //Mat imageResize = this.resize(scarface.clone(), new Size(32, 32));
                            Mat imageResize = this.resize(scarface.clone(), new Size(100, 100));
                            
                            //Imgcodecs.imwrite(file.getName(), imageResize);

                            Mat grayImage = this.convertToGray(imageResize.clone());

                            float[] hog = this.getHOGDescriptors(grayImage).toArray();

                            Sample sample = new Sample(hog, classNumber);

                            this.svmClassifier.addTrainingSample(sample);

                        }
                    }
                    
                }
                
            }
            
        }
        
        this.svmClassifier.train();
        
        System.out.println("OK");
        
    }
    
    private void writeFile(StringBuilder builder, String filePath) throws IOException {
        Files.write(Paths.get(filePath), builder.toString().getBytes());
    }
    
    private Mat resize(Mat image, Size size) {
        Mat dest = new Mat();
        Imgproc.resize(image, dest, size);
        return dest;
    }
    
    private MatOfFloat getHOGDescriptors(Mat image) {
        /*
        Size winsize = new Size(32, 32);
        Size blocksize = new Size(16, 16);
        Size blockStride = new Size(8, 8);
        Size cellsize = new Size(8, 8);
        int nbins = 9;
        */
        
        Size winsize = new Size(40, 24);
        Size blocksize = new Size(8, 8);
        Size blockStride = new Size(16, 16);
        Size cellsize = new Size(2, 2);
        int nbins = 9;
        
        HOGDescriptor hog = new HOGDescriptor(
                winsize, blocksize, blockStride, cellsize, nbins);
        
        /*
        Size winStride = new Size(8, 8);
        Size padding = new Size(0, 0);
        */
        Size winStride = new Size(16, 16);
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
    
    private File getFile(String path) throws URISyntaxException {
        return new File(getClass().getResource(path).toURI());
    }
    
    public static void main(String[] args) throws URISyntaxException, IOException {
        
        TestOCV testOCV = new TestOCV();
        testOCV.training();
        
        double[] confidences = new double[6];
        
        Mat testImage = Imgcodecs.imread(testOCV.getPath("/testocv/image/test.jpeg"));
        
        float[] hog = testOCV.getHOGDescriptors(testImage).toArray();
        
        Sample testSample = new Sample(hog);
        
        int prediction = (int) testOCV.getSvmClassifier().predict(
                testSample, confidences
        );
        
        Set<String> keys = getKeysByValue(testOCV.getClassNames(), prediction);
        
        String emotion = keys.iterator().next();
        
        System.out.println("Emotion = " + emotion);
        
    }
    
    public static <T, E> Set<T> getKeysByValue(Map<T, E> map, E value) {
        return map.entrySet()
                .stream()
                .filter(entry -> Objects.equals(entry.getValue(), value))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }
    
}
