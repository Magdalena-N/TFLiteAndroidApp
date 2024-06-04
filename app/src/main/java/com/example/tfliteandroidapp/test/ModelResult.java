package com.example.tfliteandroidapp.test;

import com.example.tfliteandroidapp.SingleInferenceResult;

import java.util.ArrayList;
import java.util.Collections;


public class ModelResult {
    public String modelName  = "";

//    public String label  = "";

//    public String inferenceTime = "";
    public int round;

    public ArrayList<SingleInferenceResult> results = new ArrayList<>();
}
