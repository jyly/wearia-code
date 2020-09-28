package com.example.gestureia;

import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

public class Siamese_model {
    public float[] sample_feature(Interpreter tflite, float[] single_feature) {
        //Runs inference in background thread
        float[][] outPuts = new float[1][];//结果分类
        tflite.run(single_feature, outPuts);
        float []final_output=outPuts[0];
        return final_output;
    }





    public int behavior_predit( Double[][] final_feature, float[] temp_finalfeature) {
        int predittag = 0;
        int datalen = final_feature.length;
        float score = 0;
        for (int i = 0; i < datalen; i++) {
            float temp = 0;
            for (int j = 0; j < 128; j++) {
                temp += ((float) (double) final_feature[i][j] - temp_finalfeature[j]) * ((float) (double) final_feature[i][j] - temp_finalfeature[j]);
            }
            temp=(float) Math.sqrt(temp);
            score += temp;
        }
        score = score / datalen;
        if (score < 0.36) {
            predittag = 1;
        }
        Log.e(">>>", "score:" + score);
        Log.e(">>>", "predittag:" + predittag);


        return predittag;
    }
}
