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
    public double[] scale_mean = null;
    public double[] scale_scale = null;
    public Interpreter ppg_tflite = null;
    public Interpreter motion_tflite = null;
    public double[][][] final_feature = null;
    private Normal_tool nortools = new Normal_tool();

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

            String[] str_sort1 = temp_sort1.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] strsort2 = temp_sort2.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_mean = temp_scale_mean.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_scale = temp_scale_scale.replace("[", "").replace("]", "").replace(" ", "").split(",");

            sort1 = nortools.strarraytointarray(str_sort1);
            sort2 = nortools.strarraytointarray(strsort2);
            scale_mean = nortools.strarraytodoublearray(str_scale_mean);
            scale_scale = nortools.strarraytodoublearray(str_scale_scale);
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
                //提取128位最终的向量
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
//                Log.e(">>>", "ppg_str_feature length:" + ppg_str_feature[0].length);

                final_feature = new double[listnum][2][128];
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < 128; j++) {
                        final_feature[i][0][j] = Double.parseDouble(ppg_str_feature[i][j]);
                    }
                }

                ArrayList<String> motion_feature = new ArrayList<String>();
                fileName = context.getExternalFilesDir("").getAbsolutePath() + "motionbasedfeature.csv";//文件存储路径
                Log.e(">>>", "motionbasedfeature filename:" + fileName);
                file = new File(fileName);
                reader = new BufferedReader(new FileReader(file));
                while ((line = reader.readLine()) != null) {
                    motion_feature.add(line);
                }
                reader.close();
                //提取128位最终的向量
                String[][] motion_str_feature = new String[listnum][];
                for (int i = 0; i < listnum; i++) {
                    motion_str_feature[i] = motion_feature.get(i).split(",");
                }
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < 128; j++) {
                        final_feature[i][1][j] = Double.parseDouble(motion_str_feature[i][j]);
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
