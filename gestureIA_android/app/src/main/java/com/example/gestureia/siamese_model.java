package com.example.gestureia;

import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

public class siamese_model {
    public float[] sample_feature(Interpreter tflite,final Double[] single_feature) {
        //Runs inference in background thread
        float[] outPutsx = new float[256];//结果分类
        tflite.run(single_feature, outPutsx);
        return outPutsx;
    }

}
