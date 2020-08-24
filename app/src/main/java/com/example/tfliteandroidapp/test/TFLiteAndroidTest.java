package com.example.tfliteandroidapp.test;

import android.app.Activity;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;

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
