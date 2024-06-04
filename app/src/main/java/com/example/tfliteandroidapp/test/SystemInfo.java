package com.example.tfliteandroidapp.test;

import android.os.Build;

public class SystemInfo{
    public String manufacturer = Build.MANUFACTURER;

    public String model= Build.MODEL;

    public String hardware = Build.HARDWARE;

    public String board  = Build.BOARD;

    public Integer apiLevel = Build.VERSION.SDK_INT;
}
