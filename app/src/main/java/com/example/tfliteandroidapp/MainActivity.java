package com.example.tfliteandroidapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import com.example.tfliteandroidapp.test.TFLiteAndroidTest;
import com.example.tfliteandroidapp.test.TFLiteAndroidTest.Device;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private TFLiteAndroidTest tfLiteAndroidTest;
    private Spinner deviceSpinner;
    private Spinner versionSpinner;
    private Spinner batchSizeSpinner;
    private Button startButton;
    private Thread tfLiteThread;
    private final int MAX_LOGS = 10;
    private ArrayList<String> logs;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        tfLiteAndroidTest = new TFLiteAndroidTest(this);
        deviceSpinner = findViewById(R.id.deviceSpinner);
        deviceSpinner.setOnItemSelectedListener(this);
        versionSpinner = findViewById(R.id.versionSpinner);
        versionSpinner.setOnItemSelectedListener(this);
        batchSizeSpinner = findViewById(R.id.batchSizeSpinner);
        batchSizeSpinner.setOnItemSelectedListener(this);
        startButton= findViewById(R.id.button);
        logs = new ArrayList<String>();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id)
    {
        if (parent == deviceSpinner) {
            Device device = Device.valueOf(parent.getItemAtPosition(pos).toString());
//            tfLiteAndroidTest.setDevice(device);
        } else if (parent == versionSpinner) {
            String version = parent.getItemAtPosition(pos).toString();
//            tfLiteAndroidTest.setVersion(version);
        } else if (parent == batchSizeSpinner) {
            String batchSize = parent.getItemAtPosition(pos).toString();
//            tfLiteAndroidTest.setBatchSize(Integer.parseInt(batchSize));
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent)
    {
    }

    public void runTestsOnClick(View view)
    {
        deviceSpinner.setEnabled(false);
        versionSpinner.setEnabled(false);
        startButton.setEnabled(false);
        batchSizeSpinner.setEnabled(false);
        updateLogs("Tests start");
        tfLiteThread = new Thread(tfLiteAndroidTest);
        tfLiteThread.start();
    }

    public void enableUI()
    {
        deviceSpinner.setEnabled(true);
        versionSpinner.setEnabled(true);
        startButton.setEnabled(true);
        batchSizeSpinner.setEnabled(true);
    }

    public void updateLogs(String log)
    {
        int size;
        TextView logView = findViewById(R.id.logs);
        logView.setText("");

        logs.add(log);
        size = logs.size();
        List<String> subList = logs.subList(size-Math.min(size,MAX_LOGS), size);
        for (String el : subList)
            logView.setText(el + "\n" + logView.getText());
    }
}