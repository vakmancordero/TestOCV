package testocv;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.commons.io.FileUtils;

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

import testocv.train.LibSVM;
import testocv.train.Sample;

/**
 *
 * @author VakSF
 */
public class TestOCV {
    
    private final CascadeClassifier classifier;
    
    private final LibSVM svmClassifier;
    
    private final Map<String, Integer> faces = initFaces();
    private final Map<String, Integer> hands = initHands();
    
    private int size;
    
    private Map<String, Integer> initFaces() {
        
        Map<String, Integer> map = new HashMap<>();
        
        map.put("Enojado",      0);
        map.put("Sorpresa",     1);
        map.put("Miedo",        2);
        map.put("Felicidad",    3);
        map.put("Triste",       4);
        map.put("Disgusto",     5);
        
        return map;
    }
    
    private Map<String, Integer> initHands() {
        
        Map<String, Integer> map = new HashMap<>();
        
        map.put("A",        0);
        map.put("B",        1);
        map.put("C",        2);
        map.put("V",        3);
        map.put("Five",     4);
        map.put("Point",    5);
        
        return map;
    }

    public Map<String, Integer> getFaces() {
        return Collections.unmodifiableMap(this.faces);
    }

    public Map<String, Integer> getHands() {
        return Collections.unmodifiableMap(hands);
    }
    
    public TestOCV(int numFeatures, int size) {
        
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        this.classifier = new CascadeClassifier(this.getPath(
                "/testocv/xml/haarcascade_frontalface_default.xml"));
        
        this.svmClassifier = new LibSVM(numFeatures);
        this.size = size;
    }

    public LibSVM getSvmClassifier() {
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
    
    public void renameFiles() throws URISyntaxException, IOException {
        
        for (File folder : this.getFile("/testocv/image").listFiles()) {
            
            if (folder.isDirectory()) {
                
                String folderName = folder.getName();
                
                File[] files = folder.listFiles();
                
                for (int i = 0; i < files.length; i++) {
                    
                    File file = files[i];
                    
                    String extension = file.getName().split("\\.(?=[^\\.]+$)")[1];
                    
                    if (extension.equalsIgnoreCase("jpg") ||
                            extension.equalsIgnoreCase("jpg") ||
                                extension.equalsIgnoreCase("jpeg")) {
                        
                        File newFile = new File(folderName.toLowerCase() + (i + 1) + ".jpg");
                        
                        FileUtils.moveFile(file, newFile);
                        
                    }
                    
                }
                
            }
            
        }
        
    }
    
    public void handTraining() throws URISyntaxException {
        
        for (File file : this.getFile("/testocv/data/train").listFiles()) {
            
            String fileName = file.getName();
            
            Mat image = Imgcodecs.imread(getPath(
                    "/testocv/data/train/" + fileName));
            
            String extension = fileName.split("\\.(?=[^\\.]+$)")[1];
            
            if (extension.equalsIgnoreCase("jpg") ||
                    extension.equalsIgnoreCase("jpg") ||
                        extension.equalsIgnoreCase("jpeg")) {
                
                if (image != null) {
                    
                    int classNumber = this.hands.get(fileName.split("-")[0]);
                    
                    System.out.println(fileName + " \t:\t " + image);
                    
                    Mat imageResize = this.resize(image.clone(), new Size(size, size));
                    
                    Mat grayImage = this.convertToGray(imageResize.clone());
                    
                    float[] hog = this.getHOGDescriptors(grayImage).toArray();
                    
                    Sample sample = new Sample(hog, classNumber);
                    
                    this.svmClassifier.addTrainingSample(sample);
                    
                }
            }
            
        }
        
        this.svmClassifier.train();
        
        System.out.println("OK Hand");
        
    }
    
    public void facesTraining() throws URISyntaxException {
        
        this.svmClassifier.load("save.bin");
        
//        for (File folder : this.getFile("/testocv/image").listFiles()) {
//            
//            if (folder.isDirectory()) {
//                
//                String folderName = folder.getName();
//                
//                if (!folderName.equalsIgnoreCase("training")) {
//                    
//                    int classNumber = this.faces.get(folderName);
//
//                    System.out.println(folderName + " : " + classNumber + "\n");
//
//                    for (File file : folder.listFiles()) {
//
//                        Mat scarface = this.getScarface(
//                                "/testocv/image/" + folderName + "/" + file.getName());
//
//                        String extension = file.getName().split("\\.(?=[^\\.]+$)")[1];
//
//                        if (extension.equalsIgnoreCase("jpg") || 
//                                extension.equalsIgnoreCase("jpg") ||
//                                    extension.equalsIgnoreCase("jpeg")) {
//
//                            if (scarface != null) {
//
//                                System.out.println(file.getName() + " \t:\t " + scarface);
//
//                                Mat imageResize = this.resize(scarface.clone(), new Size(size, size));
//                                Mat grayImage = this.convertToGray(imageResize.clone());
//
//                                float[] hog = this.getHOGDescriptors(grayImage).toArray();
//                                
//                                System.out.println("size = " + hog.length);
//
//                                Sample sample = new Sample(hog, classNumber);
//
//                                this.svmClassifier.addTrainingSample(sample);
//
//                            }
//                        }
//
//                    }
//                }
//                
//            }
//            
//        }
//        
//        this.svmClassifier.train();
//        
//        this.svmClassifier.save("save.bin");
        
        System.out.println("OK Face");
        
    }
    
    private void writeFile(StringBuilder builder, String filePath) throws IOException {
        Files.write(Paths.get(filePath), builder.toString().getBytes());
    }
    
    public Mat resize(Mat image, Size size) {
        Mat dest = new Mat();
        Imgproc.resize(image, dest, size);
        return dest;
    }
    
    public MatOfFloat getHOGDescriptors(Mat image) {
        
        /*
        Size winsize = new Size(32, 32);
        Size blocksize = new Size(16, 16);
        Size blockStride = new Size(8, 8);
        Size cellsize = new Size(8, 8);
        int nbins = 9;
        */
        
        Size winsize = new Size(64, 64);
        Size blocksize = new Size(32, 32);
        Size blockStride = new Size(16, 16);
        Size cellsize = new Size(16, 16);
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
    
    public Mat getScarface(String fileName) {
        
        Mat image = Imgcodecs.imread(getPath(fileName));
        MatOfRect facesRect = new MatOfRect();
        
        this.classifier.detectMultiScale(image, facesRect);
        
        List<Rect> faces = new ArrayList<>(Arrays.asList(facesRect.toArray()));
        
        return !faces.isEmpty() ? 
                new Mat(image, faces.iterator().next()) : null;
        
    }
    
    public Mat convertToGray(Mat mat) {
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY); return mat;
    }
    
    public String getPath(String url) {
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
        
    }
    
    public static <T, E> Set<T> getKeysByValue(Map<T, E> map, E value) {
        return map.entrySet()
                .stream()
                .filter(entry -> Objects.equals(entry.getValue(), value))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }
    
}
