package com.example.tfliteandroidapp.test;

import android.app.Activity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;

public class TFLiteAndroidTest {

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
    private Activity activity;

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

    public TFLiteAndroidTest(Activity pA)
    {
        activity = pA;
        currentDevice = Device.CPU;
        gpuDelegate = null;
        nnApiDelegate = null;
        tfliteOptions = new Interpreter.Options();
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

        if (isQuantized)
            probSTD = 255;
        else
            probSTD = 1;

        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0.0f, probSTD)).build();
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
}
