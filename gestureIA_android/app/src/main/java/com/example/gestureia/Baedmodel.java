package com.example.gestureia;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

public class Baedmodel {
    public int[] sort1 = null;
    public int[] sort2 = null;
    public float[] scale_mean = null;
    public float[] scale_scale = null;
    public Interpreter ppg_tflite = null;
    public Interpreter motion_tflite = null;
    public float[][][] final_feature = null;

    public void readmodelpara(Context context) {
        try {
            ppg_tflite = new Interpreter(loadModelFile(context,"ppg_based_model"));
            motion_tflite = new Interpreter(loadModelFile(context,"motion_based_model"));
            InputStream parameterinput = context.getAssets().open("stdpropara.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(parameterinput));
            String temp_sort1 = reader.readLine();
            String temp_sort2 = reader.readLine();
            String temp_scale_mean = reader.readLine();
            String temp_scale_scale = reader.readLine();
            reader.close();
            parameterinput.close();
            reader=null;
            parameterinput=null;

            String[] str_sort1 = temp_sort1.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] strsort2 = temp_sort2.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_mean = temp_scale_mean.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_scale = temp_scale_scale.replace("[", "").replace("]", "").replace(" ", "").split(",");
            temp_sort1=null;
            temp_sort2=null;
            temp_scale_mean=null;
            temp_scale_scale=null;
            Normal_tool nortools = new Normal_tool();
            sort1 = nortools.strarraytointarray(str_sort1);
            sort2 = nortools.strarraytointarray(strsort2);
            scale_mean = nortools.strarraytofloatarray(str_scale_mean);
            scale_scale = nortools.strarraytofloatarray(str_scale_scale);
            nortools=null;
            str_sort1=null;
            strsort2=null;
            str_scale_mean=null;
            str_scale_scale=null;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Context context,String model) throws IOException {
        Log.e(">>>", model + ".tflite");
        AssetFileDescriptor fileDescriptor = context.getApplicationContext().getAssets().openFd(model + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public void readbasedfeature(Context context) {
        try {
            ArrayList<String> ppg_feature = new ArrayList<String>();
            String fileName = context.getExternalFilesDir("").getAbsolutePath() + "ppgbasedfeature.csv";//文件存储路径
            Log.e(">>>", "ppgbasedfeature filename:" + fileName);
            File file = new File(fileName);
            if (file.exists()) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = reader.readLine()) != null) {
                    ppg_feature.add(line);
                }
                reader.close();
                reader=null;
                file=null;
                //提取注册样本的向量
                int listnum = 0;
                if (ppg_feature.size() > 5) {
                    listnum = 5;
                } else {
                    listnum = ppg_feature.size();
                }
                String[][] ppg_str_feature = new String[listnum][];
                for (int i = 0; i < listnum; i++) {
                    ppg_str_feature[i] = ppg_feature.get(i).split(",");
                }
                ppg_feature=null;

                final_feature = new float[listnum][2][ppg_str_feature[0].length];
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < ppg_str_feature[0].length; j++) {
                        final_feature[i][0][j] = Float.parseFloat(ppg_str_feature[i][j]);
                    }
                }
                ppg_str_feature=null;

                ArrayList<String> motion_feature = new ArrayList<String>();
                fileName = context.getExternalFilesDir("").getAbsolutePath() + "motionbasedfeature.csv";//文件存储路径
                Log.e(">>>", "motionbasedfeature filename:" + fileName);
                file = new File(fileName);
                reader = new BufferedReader(new FileReader(file));
                while ((line = reader.readLine()) != null) {
                    motion_feature.add(line);
                }
                reader.close();
                reader=null;
                file=null;
                String[][] motion_str_feature = new String[listnum][];
                for (int i = 0; i < listnum; i++) {
                    motion_str_feature[i] = motion_feature.get(i).split(",");
                    for (int j = 0; j < motion_str_feature[i].length; j++) {
                        final_feature[i][1][j] = Float.parseFloat(motion_str_feature[i][j]);
                    }
                }
                motion_feature=null;
                motion_str_feature=null;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //将注册样本特征写入文件中
    public void writebasedfeature(Context context,ArrayList<float[][]> featureset) {
        int arraylength = featureset.size();
        int featurelen = featureset.get(0)[0].length;
        try {
            String fileName = context.getExternalFilesDir("").getAbsolutePath() + "ppgbasedfeature.csv";//文件存储路径
            Log.e(">>>","filename:"+fileName);
            File file=new File(fileName);
            if(file.exists()){
                file.delete();
                file.createNewFile();
            }
            BufferedWriter out = new BufferedWriter(new FileWriter(file));
            for (int i = 0; i < arraylength; i++) {
                for (int j = 0; j < featurelen; j++) {
                    out.write(featureset.get(i)[0][j] + ",");
                }
                out.newLine();
            }
            out.close();

            fileName = context.getExternalFilesDir("").getAbsolutePath() + "motionbasedfeature.csv";//文件存储路径
            Log.e(">>>","filename:"+fileName);
            file=new File(fileName);
            if(file.exists()){
                file.delete();
                file.createNewFile();
            }
            out = new BufferedWriter(new FileWriter(file));
            for (int i = 0; i < arraylength; i++) {
                for (int j = 0; j < featurelen; j++) {
                    out.write(featureset.get(i)[1][j] + ",");
                }
                out.newLine();
            }
            out.close();
            out=null;
            file=null;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //将特征转化为通过网络后的特征
    public float[][] dataprocess(double[] featureset){
        int featurelen = 30;
        float[] inform_feature = new float[featurelen * 2];
        for (int i = 0; i < featurelen; i++) {
            inform_feature[i] = (float)  featureset[sort1[i]];
            inform_feature[i + featurelen] = (float)  featureset[sort2[i] + 76];
        }
//                            inform_feature = featurecontrol.featurestd(inform_feature, basedmodel.scale_mean, basedmodel.scale_scale);
        float[] ppg_feature = new float[featurelen];
        float[] motion_feature = new float[featurelen];
        for (int i = 0; i < featurelen; i++) {
            ppg_feature[i] = inform_feature[i];
            motion_feature[i] = inform_feature[i + featurelen];
        }

        float[][] final_feature = new float[2][];
        final_feature[0] = sample_feature(ppg_tflite, ppg_feature);
        final_feature[1] = sample_feature(motion_tflite, motion_feature);
        inform_feature=null;
        ppg_feature=null;
        motion_feature=null;
        return final_feature;
    }

    public float[] sample_feature(Interpreter tflite, float[] single_feature) {
        float[][] outPuts = new float[1][32];//结果分类
        tflite.run(single_feature, outPuts);
        float[] final_output = outPuts[0];
//        for (int i = 0; i < final_output.length; i++) {
//            System.out.print(final_output[i]+",");
//        }
//        System.out.println("");
        return final_output;
    }


    public int behavior_predit(float[][][] final_feature, float[][] temp_final_feature) {
        int predittag = 0;
        int datalen = final_feature.length;
        float score = 0;
        for (int i = 0; i < datalen; i++) {
            float temp1 = 0;
            for (int j = 0; j < final_feature[0][0].length; j++) {
                temp1 += (final_feature[i][0][j] - temp_final_feature[0][j]) * (final_feature[i][0][j] - temp_final_feature[0][j]);
            }
            temp1 = (float) Math.sqrt(temp1);
            float temp2 = 0;
            for (int j = 0; j < final_feature[0][0].length; j++) {
                temp2 += (final_feature[i][1][j] - temp_final_feature[1][j]) * (final_feature[i][1][j] - temp_final_feature[1][j]);
            }
            temp2 = (float) Math.sqrt(temp2);
//            score += (temp1 + temp2) / 2;
            score += temp1;
        }
        score = score / datalen;
        if (score < 0.5) {
            predittag = 1;
        }
        Log.e(">>>", "score:" + score);
        Log.e(">>>", "predittag:" + predittag);
        return predittag;
    }
}
