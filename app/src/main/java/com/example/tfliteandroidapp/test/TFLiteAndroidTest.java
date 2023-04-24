package com.example.tfliteandroidapp.test;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;

import com.example.tfliteandroidapp.MainActivity;
import com.example.tfliteandroidapp.R;
import com.opencsv.CSVWriter;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TFLiteAndroidTest implements Runnable {

    private static final int NUMBER_OF_IMAGE_SAMPLES = 1;
    private static final int MAX_RESULTS = 5;
    private int i;

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    public enum UIUpdate {
        PRINT_MSG,
        ENABLE_UI
    }

    /** TFLite model loaded into memory */
    private MappedByteBuffer tfliteModel;

    /** Interpreter which runs model inference with Tensorflow Lite. */
    private Interpreter interpreter;

    /** Options of interpreter */
    private Interpreter.Options tfliteOptions;

    /** Activity in which the test is performed */
    private MainActivity activity;

    /** Current device used for executing classification */
    private Device currentDevice;

    /** Optional GPU delegate  */
    private GpuDelegate gpuDelegate;

    /** Optional NNAPI delegate */
    private NnApiDelegate nnApiDelegate;

    /** Input image TensorBuffer of current interpreter*/
    private TensorImage[] inputImageBuffers;
    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer of current interpreter*/
    private TensorBuffer outputProbabilityBuffer;

    /** Processer to apply post processing of the output probability */
    private TensorProcessor probabilityProcessor;

    /** Shape of input image */
    private int imageSizeY, imageSizeX;

    private int batchSize;

    private float imgMean, imgStd;

    private float probMean = 0.0f, probStd;

    /** Writer to file with output results*/
    CSVWriter csvWriter;

    /** Name of directory with models inside assets/models*/
    String modelsDir;

    String baseDir;

    public TFLiteAndroidTest(MainActivity pA)
    {
        activity = pA;
        currentDevice = Device.CPU;
        gpuDelegate = null;
        nnApiDelegate = null;
        tfliteOptions = new Interpreter.Options();
        modelsDir = "mobilenet_v1";
        batchSize = 2;
        baseDir = "models/";
    }

    /**
     * Main method which runs 330 inferences for each model.
     * We have 33 datasets and from each dataset we get 10 randomly
     * chosen images.
     */
    public void run()
    {
        List<String> models;
        String[] dataSets;
        List<Bitmap> images;
        List<String> labels = null;
        long startTime, endTime;
        String modelName;
        int i, j;
        ByteBuffer inputBuffer;
        inputImageBuffers = new TensorImage[batchSize];

        models = getListOfModels();

        if (models == null)
            return;

        dataSets = activity.getResources().getStringArray(R.array.datasets);
        try {
            labels = FileUtil.loadLabels(activity, "labels.txt");
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        if (batchSize == 1)
            prepareWriter("results_" + modelsDir + "_" + currentDevice + ".csv");
        else
            prepareWriter("results_" + modelsDir + "_" + currentDevice + "_bs" + batchSize + ".csv");

        for (String model : models) {
            if (currentDevice == Device.GPU && model.contains("quant")) {
                updateUI(UIUpdate.PRINT_MSG,"GPU doesn't support quantized models");
                continue;
            }
            if (model.contains("edgetpu") && batchSize > 1)
                continue;
            initInterpreter(baseDir + modelsDir + "/" + model);
            prepareBuffers();
            modelName = model.replace(".tflite","");
            updateUI(UIUpdate.PRINT_MSG,"Model loaded: " + modelName);

            dataSets = Arrays.copyOfRange(dataSets, 0, 1);
            for (String dataSet : dataSets) {
                String[] dataSetInfo = getLabelAndURL(dataSet);
                images = getImagesFromDir("datasets/" + dataSetInfo[0]);
                updateUI(UIUpdate.PRINT_MSG,"DataSet loaded: " + dataSetInfo[0]);
//                for (i = 0; i < images.size(); i += batchSize) {
                for(Bitmap image: images){

                    inputImageBuffer = processImage(image);
//                    processImage(images.subList(i, i + batchSize).toArray());
//                    inputBuffer = ByteBuffer.allocate(inputImageBuffers[0].getBuffer().capacity()*batchSize);
//                    for (j = 0; j < batchSize; j++)
//                        inputBuffer.put(inputImageBuffers[j].getBuffer());
//                    inputBuffer.order(ByteOrder.nativeOrder());
                    startTime = SystemClock.uptimeMillis();
                    try{
                        interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                        endTime = SystemClock.uptimeMillis();
                        
                        double t = (double)endTime - (double)startTime;
                        if (batchSize == 1) {
                            Map<String, Float> labeledProbability =
                                    new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                                            .getMapWithFloatValue();
                            Map.Entry<String, Float> max = Collections.max(labeledProbability.entrySet(),
                                    (Map.Entry<String, Float> e1, Map.Entry<String, Float> e2) -> e1.getValue().compareTo(e2.getValue()));
                            saveKBestResults(labeledProbability);
                        } else {
                            Map<String, Float> labeledProbability =
                                    new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                                            .getMapWithFloatValue();
                            Map.Entry<String, Float> max = Collections.max(labeledProbability.entrySet(),
                                    (Map.Entry<String, Float> e1, Map.Entry<String, Float> e2) -> e1.getValue().compareTo(e2.getValue()));
                            saveKBestResults(labeledProbability);
//                            saveKDummyResults();
                        }
                    }
                    catch (Exception e){
                        saveResult(modelName, dataSetInfo[0], "Model did not complete the inference");
                        e.printStackTrace();
                    }
                }
            }
        }
        try {
            csvWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        updateUI(UIUpdate.PRINT_MSG, "DONE");
        updateUI(UIUpdate.ENABLE_UI, null);
    }

    /**
     * Initiates interpreter, loads model, optionally adds delegate
     *
     * @param model Name of .tflite file
     */
    private void initInterpreter(String model)
    {
        tfliteOptions = new Interpreter.Options();
        switch (currentDevice) {
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case NNAPI:
                nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case CPU:
                break;
        }
        try {
            tfliteModel = FileUtil.loadMappedFile(activity, model);
            System.out.println(model);
            interpreter = new Interpreter(tfliteModel, tfliteOptions);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (model.contains("quant")) {
            probStd = 255.0f;
            imgMean = 0.0f;
            imgStd = 1.0f;
        }
        else {
            probStd = 1.0f;
            imgMean = 127.5f;
            imgStd = 127.5f;
        }
    }

    /**
     * Reads image shape, image data type from InputTensor and probability shape
     * and probability data type from OutputTensor. Initiates inputImageBuffer and
     * outputProbabilityBuffer.
     *
     * Initiates probabilityProcessor with mean equal to 0 and std equal to 255 when mobilenet
     * model is quantized because such model needs additional dequantization to the output
     * probability. When model is not quantized then std is set to 1.
     */
    private void prepareBuffers()
    {
        int[] imageShape;
        int[] probabilityShape;
        DataType imageDataType;
        DataType probabilityDataType;
        float probSTD;

        imageShape = interpreter.getInputTensor(0).shape();
        imageShape[0] = batchSize;
//        interpreter.resizeInput(0 , imageShape);
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        imageDataType = interpreter.getInputTensor(0).dataType();
        probabilityShape = interpreter.getOutputTensor(0).shape();
        probabilityDataType = interpreter.getOutputTensor(0).dataType();
        System.out.println(imageDataType);
        inputImageBuffer = new TensorImage(imageDataType);
//        for (int i = 0; i < batchSize; i++)
//            inputImageBuffers[i] = new TensorImage(imageDataType);
//
//        probabilityShape[0] *= batchSize;
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(probMean, probStd)).build();
    }

    private TensorImage processImage(final Bitmap bitmap)
    {
        int cropSize;

        inputImageBuffer.load(bitmap);

        cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new NormalizeOp(imgMean, imgStd))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    /**
     * Performs image loading, croping and resizing.
     *
     * @param bitmaps Bitmap of image to recognize
     * @return TensorImage for model's input
     */
    private void processImage(final Object[] bitmaps)
    {
        int cropSize, i;
        Bitmap bitmap;
        for(i = 0; i < batchSize; i++) {
            bitmap = (Bitmap) bitmaps[i];
            inputImageBuffers[i].load(bitmap);
            cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                            .add(new ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                            .add(new NormalizeOp(imgMean, imgStd))
                            .build();

            inputImageBuffers[i] = imageProcessor.process(inputImageBuffers[i]);
        }
    }

    /**
     * Returns list of models (filenames with ext.) available in models folder.
     *
     * @return List of models or null in case of exception
     */
    private List<String> getListOfModels()
    {
        try {
            return Arrays.asList(activity.getAssets().list(baseDir + modelsDir));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    /**
     * Splits string from array of datasets to label and url to dataset.
     *
     * @return Array of strings which contains label at index 0 and url at index 1.
     */
    private String[] getLabelAndURL(String str)
    {
        return str.split(" ");
    }

    /**
     * Initiates CSVWriter, output file is located in
     * /data/data/<package-name>/files/
     *
     * @param outputFileName Name of csv file with results.
     */
    private void prepareWriter(String outputFileName)
    {
        try {
            csvWriter = new CSVWriter(new OutputStreamWriter(activity.openFileOutput(outputFileName, Context.MODE_PRIVATE)));
            String[] header = { "ModelName", "Label", "InferenceTime", "Recognition", "Accuracy" };
            csvWriter.writeNext(header);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    /**
     * Returns a list of images urls from the data set URL.
     * Returns null in case of an exception.
     *
     * @param datasetURLString URL of the data set.
     * @return list of images URLs or null
     */
    private List<String> getImagesURLList(String datasetURLString){
        ArrayList<String> imagesList = null;
        String inputLine;
        try {
            URL datasetURL = new URL(datasetURLString);
            try(BufferedReader input = new BufferedReader(new InputStreamReader(datasetURL.openStream()))){
                imagesList = new ArrayList<>();
                while((inputLine = input.readLine()) != null){
                    imagesList.add(inputLine);
                }
            }
            catch (IOException e){
                /* pass */
                e.printStackTrace();
            }
        } catch (MalformedURLException e){
            /* pass */
            e.printStackTrace();
        }
        return imagesList;
    }

    /**
     * Returns a list of bitmaps randomly chosen from urls given.
     * Returns null in case of an exception.
     *
     * @param imagesURLs list of URLs as Strings
     * @param numOfImages number of images to get. In case numOfImages > imagesURLs.size()
     *                   or numOfImages == 0 downloads the whole data set
     * @return list of bitmaps or null
     */
    private List<Bitmap> getImagesBitmapsList(List<String> imagesURLs, int numOfImages){
        ArrayList<Bitmap> imagesList = new ArrayList<>();
        int imagesAdded = 0;
        Collections.shuffle(imagesURLs);
        for (String url : imagesURLs){
            try {
                URL imageURL = new URL(url);
                Bitmap image = BitmapFactory.decodeStream(imageURL.openStream());
                if (image != null){
                    imagesList.add(image);
                    if (++imagesAdded >= numOfImages)
                        break;
                }
            }
            catch (IOException e) {
                /* pass */
            }
        }
        return imagesList;
    }

    /**
     * Returns a list of bitmaps randomly chosen from directory given.
     * Returns null in case of an exception.
     *
     * @param path path to directory with images
     * @return list of bitmaps or null
     */
    private List<Bitmap> getImagesFromDir(String path) {
        ArrayList<Bitmap> imagesList = new ArrayList<>();
        int i = 0;
        int numSamples = NUMBER_OF_IMAGE_SAMPLES;

        switch (batchSize) {
            case 4:
                numSamples = 12;
                break;
            case 8:
            case 16:
                numSamples = 16;
                break;
            case 32:
                numSamples = 32;
                break;
        }
        try {
            ArrayList<String> list =  new ArrayList<String>(Arrays.asList(activity.getAssets().list(path)));
            Collections.shuffle(list);

            for (String imageFile : list) {
                imagesList.add(BitmapFactory.decodeStream(activity.getAssets().open(path + "/" + imageFile)));
                i++;
                if (i == numSamples)
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return imagesList;
    }

    /**
     * Saves top K results of inference in csv file
     *
     * @param labeledProbability results of inference.
     */
    private void saveKBestResults(Map<String, Float> labeledProbability) {
        for (int i = 1; i <= 5; i++) {
            Map.Entry<String, Float> max = Collections.max(labeledProbability.entrySet(),
                    (Map.Entry<String, Float> e1, Map.Entry<String, Float> e2) -> e1.getValue().compareTo(e2.getValue()));
            String[] str = {"", "", "", max.getKey(), Float.toString(max.getValue())};
            csvWriter.writeNext(str);
            updateUI(UIUpdate.PRINT_MSG, i + "." + max.getKey() + ": " + max.getValue());
            labeledProbability.replace(max.getKey(), 0.0f);
        }

    }

    private void saveKDummyResults() {
        for (int i = 1; i <= 5; i++) {
            String[] str = {"", "", "", "", ""};
            csvWriter.writeNext(str);
            updateUI(UIUpdate.PRINT_MSG, i + "." + "" + ": " + "");
        }

    }

    /**
     * Saves result of one inference in csv file
     *
     * @param modelName Name of tested model.
     * @param label correct answer
     * @param inferenceTime time of inference
     */
    private void saveResult(String modelName, String label, String inferenceTime)
    {
        String[] str = {modelName, label, inferenceTime, "", ""};
        csvWriter.writeNext(str);
        updateUI(UIUpdate.PRINT_MSG, modelName + " Time: " + inferenceTime + " Label: " + label);
    }

    public void setDevice(Device device)
    {
        currentDevice = device;
    }

    public void setVersion(String v) {
        modelsDir = v;
    }

    public void setBatchSize(int bS) {
        batchSize = bS;

        if (batchSize > 1)
            baseDir = "models/converted_models_batch/";
        else
            baseDir = "models/converted_models_batch/";
    }
    /**
     * Update UI from main thread
     *
     * @param uiUpdate enum which indicates what should be perform on UI thread.
     * @param msg optional message
     */
    private void updateUI(UIUpdate uiUpdate, String msg)
    {
        activity.runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        switch (uiUpdate) {
                            case PRINT_MSG:
                                activity.updateLogs(msg);
                                break;
                            case ENABLE_UI:
                                activity.enableUI();
                                break;
                            default:
                        }
                    }
                }
        );
    }
}
