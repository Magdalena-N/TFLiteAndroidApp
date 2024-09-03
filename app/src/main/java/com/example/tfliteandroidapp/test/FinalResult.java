package com.example.tfliteandroidapp.test;

import com.google.firebase.firestore.FieldValue;

import java.util.ArrayList;


public class FinalResult {
    public SystemInfo systemInfo = new SystemInfo();

    public ArrayList<ModelResult> modelResults = new ArrayList<>();

    public FieldValue createdAt;

}
