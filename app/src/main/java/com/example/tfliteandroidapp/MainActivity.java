package com.example.tfliteandroidapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Spinner;

import com.example.tfliteandroidapp.test.TFLiteAndroidTest;
import com.example.tfliteandroidapp.test.TFLiteAndroidTest.Device;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private TFLiteAndroidTest tfLiteAndroidTest;
    private Spinner deviceSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tfLiteAndroidTest = new TFLiteAndroidTest(this);
        deviceSpinner = findViewById(R.id.deviceSpinner);
        deviceSpinner.setOnItemSelectedListener(this);
        Thread tfLiteThread = new Thread(tfLiteAndroidTest);
        tfLiteThread.start();
    }

    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        if (parent == deviceSpinner) {
            Device device = Device.valueOf(parent.getItemAtPosition(pos).toString());
            tfLiteAndroidTest.setDevice(device);
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    }
}