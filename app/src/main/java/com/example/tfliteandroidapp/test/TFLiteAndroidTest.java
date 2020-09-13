package com.example.tfliteandroidapp.test;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

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
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class TFLiteAndroidTest implements Runnable {

    private static final int NUMBER_OF_IMAGE_SAMPLES = 10;

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /** TFLite model loaded into memory */
    private MappedByteBuffer tfliteModel;

    /** Interpreter which runs model inference with Tensorflow Lite. */
    private Interpreter interpreter;

    /** Options of interpreter */
    private final Interpreter.Options tfliteOptions;

    /** Activity in which the test is performed */
    private MainActivity activity;

    /** Current device used for executing classification */
    private Device currentDevice;

    /** Optional GPU delegate  */
    private GpuDelegate gpuDelegate;

    /** Optional NNAPI delegate */
    private NnApiDelegate nnApiDelegate;

    /** Input image TensorBuffer of current interpreter*/
    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer of current interpreter*/
    private TensorBuffer outputProbabilityBuffer;

    /** Processer to apply post processing of the output probability */
    private TensorProcessor probabilityProcessor;

    /** Shape of input image */
    private int imageSizeY, imageSizeX;

    private boolean isQuantized;

    private float imgMean, imgStd;

    private float probMean = 0.0f, probStd;

    /** Writer to file with output results*/
    CSVWriter csvWriter;

    public TFLiteAndroidTest(MainActivity pA)
    {
        activity = pA;
        currentDevice = Device.CPU;
        gpuDelegate = null;
        nnApiDelegate = null;
        tfliteOptions = new Interpreter.Options();
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

        models = getListOfModels();

        if (models == null)
            return;

        dataSets = activity.getResources().getStringArray(R.array.datasets);
        prepareWriter("results.csv");

        for (String model : models) {
            initInterpreter(model);
            prepareBuffers();

            for (String dataSet : dataSets) {
                /**
                 * TODO
                 * Get 10 images from dataSet
                 */
                String[] dataSetInfo = getLabelAndURL(dataSet);
                images = getImagesBitmapsList(getImagesURLList(dataSetInfo[1]), NUMBER_OF_IMAGE_SAMPLES);
//                for (Bitmap image : images) {
                    /**
                     * TODO
                     * Run inferences and save results
                     */
                    //saveResult(model,TODO);
//                }
            }
        }
        try {
            csvWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        activity.runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        activity.enableSpinner();
                    }
                }
        );
    }

    /**
     * Initiates interpreter, loads model, optionally adds delegate
     *
     * @param model Name of .tflite file
     */
    private void initInterpreter(String model)
    {
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
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        imageDataType = interpreter.getInputTensor(0).dataType();
        probabilityShape = interpreter.getOutputTensor(0).shape();
        probabilityDataType = interpreter.getOutputTensor(0).dataType();

        inputImageBuffer = new TensorImage(imageDataType);
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(probMean, probStd)).build();
    }

    /**
     * Performs image loading, croping and resizing.
     *
     * @param bitmap Bitmap of image to recognize
     * @return TensorImage for model's input
     */
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
     * Returns list of models (filenames with ext.) available in models folder.
     *
     * @return List of models or null in case of exception
     */
    private List<String> getListOfModels()
    {
        try {
            return Arrays.asList(activity.getAssets().list("models"));
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
            String[] header = { "ModelName", "Accuracy", "InferenceTime", "Recognition", "Label" };
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
     * Saves result of one inference in csv file
     *
     * @param modelName Name of tested model.
     * @param accuracy accuracy of inference
     * @param inferenceTime time of inference
     * @param recognition recognized label
     * @param label correct answer
     */
    private void saveResult(String modelName, String accuracy, String inferenceTime, String recognition,
                            String label)
    {
        String[] str = {modelName, accuracy, inferenceTime, recognition, label};
        csvWriter.writeNext(str);
    }

    public void setDevice(Device device)
    {
        currentDevice = device;
    }
}
