package com.example.gestureia;

import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

public class Siamese_model {
    public float[] sample_feature(Interpreter tflite, float[] single_feature) {
        //Runs inference in background thread
        float[][] outPuts = new float[1][128];//结果分类
        tflite.run(single_feature, outPuts);
        float[] final_output = outPuts[0];
//        for (int i = 0; i < final_output.length; i++) {
//            System.out.print(final_output[i]+",");
//        }
//        System.out.println("");

        return final_output;
    }


    public int behavior_predit(double[][][] final_feature, float[][] temp_final_feature) {
        int predittag = 0;
        int datalen = final_feature.length;
        float score = 0;
        for (int i = 0; i < datalen; i++) {
            float temp1 = 0;
            for (int j = 0; j < 128; j++) {
                temp1 += ((float) (double) final_feature[i][0][j] - temp_final_feature[0][j]) * ((float) (double) final_feature[i][0][j] - temp_final_feature[0][j]);
            }
            temp1 = (float) Math.sqrt(temp1);
            float temp2 = 0;
            for (int j = 0; j < 128; j++) {
                temp2 += ((float) (double) final_feature[i][1][j] - temp_final_feature[1][j]) * ((float) (double) final_feature[i][1][j] - temp_final_feature[1][j]);
            }
            temp2 = (float) Math.sqrt(temp2);

//            score += (temp1 + temp2) / 2;
            score +=temp1 ;
        }
        score = score / datalen;
        if (score < 0.2) {
            predittag = 1;
        }
        Log.e(">>>", "score:" + score);
        Log.e(">>>", "predittag:" + predittag);


        return predittag;
    }
}
