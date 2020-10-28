package com.example.gestureia;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
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
    public Interpreter basedmodel_tflite = null;
    public float[][][] mul_final_feature = null;
    public float[][] single_final_feature = null;



    //将模型加载入系统
    public void readmodelpara(Context context) {
        try {
            ppg_tflite = new Interpreter(loadModelFile(context,"ppg_based_model"));
            motion_tflite = new Interpreter(loadModelFile(context,"motion_based_model"));
            basedmodel_tflite = new Interpreter(loadModelFile(context,"based_model"));
//            InputStream based_model = context.getAssets().open("based_model.tflite");
//            String fileName = context.getExternalFilesDir("").getAbsolutePath() + "based_model.tflite";//文件存储路径
//            Log.e(">>>", "based_model.tflite filename:" + fileName);
//            File files = new File(fileName);
//            inputStream2File(based_model,files);
//            basedmodel_tflite =  new Interpreter(files);

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
            System.gc();
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

    //读取双模型需要的特征
    public void readmulfeature(Context context) {
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

                mul_final_feature = new float[listnum][2][ppg_str_feature[0].length];
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < ppg_str_feature[0].length; j++) {
                        mul_final_feature[i][0][j] = Float.parseFloat(ppg_str_feature[i][j]);
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
                        mul_final_feature[i][1][j] = Float.parseFloat(motion_str_feature[i][j]);
                    }
                }
                motion_feature=null;
                motion_str_feature=null;
                System.gc();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //单特征处理
    public void readfeature(Context context) {
        try {
            ArrayList<String> feature = new ArrayList<String>();
            String fileName = context.getExternalFilesDir("").getAbsolutePath() + "basedfeature.csv";//文件存储路径
            Log.e(">>>", "basedfeature filename:" + fileName);
            File file = new File(fileName);
            if (file.exists()) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = reader.readLine()) != null) {
                    feature.add(line);
                }
                reader.close();
                reader=null;
                file=null;
                //提取注册样本的向量
                int listnum = 0;
                if (feature.size() > 5) {
                    listnum = 5;
                } else {
                    listnum = feature.size();
                }
                String[][] str_feature = new String[listnum][];
                for (int i = 0; i < listnum; i++) {
                    str_feature[i] = feature.get(i).split(",");
                }
                feature=null;
                single_final_feature = new float[listnum][str_feature[0].length];
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < str_feature[0].length; j++) {
                        single_final_feature[i][j] = Float.parseFloat(str_feature[i][j]);
                    }
                }
                str_feature=null;
                System.gc();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    //将特征转化为通过网络后的特征
    public float[][] featureprocess(double[] featureset){
        int featurelen = 30;
        //根据序号提取特征
        float[] inform_feature = new float[featurelen * 2];
        for (int i = 0; i < featurelen; i++) {
            inform_feature[i] = (float)  featureset[sort1[i]];
            inform_feature[i + featurelen] = (float)  featureset[sort2[i] + 76];
        }
        //对特征做标准化处理
        Featurecontrol featurecontrol=new Featurecontrol();
        inform_feature = featurecontrol.featurestd(inform_feature, scale_mean, scale_scale);
        featurecontrol=null;
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
        System.gc();
        return final_feature;
    }

    //将特征转化为通过网络后的特征
    public float[] dataprocess(double[][] madata){
        float [][][][]input=new float[1][2][300][1];
        for(int i=0;i<2;i++){
            for(int j=0;j<300;j++){
                input[0][i][j][0]=(float)madata[i][j];
            }
        }
        float[] final_feature = sample_feature(basedmodel_tflite, input);
        input=null;
        System.gc();
        return final_feature;
    }

    public void writefeature(Context context,ArrayList<float[]> featureset,String filename) {
        int arraylength = featureset.size();
        int featurelen = featureset.get(0).length;
        try {
            String fileName = context.getExternalFilesDir("").getAbsolutePath() + filename;//文件存储路径
            Log.e(">>>","filename:"+fileName);
            File file=new File(fileName);
            if(file.exists()){
                file.delete();
                file.createNewFile();
            }
            BufferedWriter out = new BufferedWriter(new FileWriter(file));
            for (int i = 0; i < arraylength; i++) {
                for (int j = 0; j < featurelen; j++) {
                    out.write(featureset.get(i)[j] + ",");
                }
                out.newLine();
            }
            out.close();
            out=null;
            file=null;
            System.gc();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //将注册样本特征写入文件中
    public void writemulfeature(Context context,ArrayList<float[][]> featureset) {
        ArrayList<float[]> x=new ArrayList<float[]>();
        ArrayList<float[]> y=new ArrayList<float[]>();
        for(int i=0;i<featureset.size();i++){
            x.add(featureset.get(i)[0]);
            y.add(featureset.get(i)[1]);
        }
        writefeature(context,x,"ppgbasedfeature.csv");
        writefeature(context,y,"motionbasedfeature.csv");
        x=null;
        y=null;
        System.gc();
    }

//    多信道特征
    public float[] sample_feature(Interpreter tflite, float[][][][] single_feature) {
        float[][] outPuts = new float[1][128];//结果分类
        tflite.run(single_feature, outPuts);
        float[] final_output = outPuts[0];
        outPuts=null;
        return final_output;
    }
//    单信道特征
    public float[] sample_feature(Interpreter tflite, float[] single_feature) {
        float[][] outPuts = new float[1][128];//结果分类
        tflite.run(single_feature, outPuts);
        float[] final_output = outPuts[0];
        outPuts=null;
        return final_output;
    }
//    多维分数
    public int behavior_predit( float[][] final_feature) {
        int predittag = 0;
        float score = 0;
        for (int i = 0; i < mul_final_feature.length; i++) {
            float temp1 = 0;
            for (int j = 0; j < mul_final_feature[0][0].length; j++) {
                temp1 += (mul_final_feature[i][0][j] - final_feature[0][j]) * (mul_final_feature[i][0][j] - final_feature[0][j]);
            }
            temp1 = (float) Math.sqrt(temp1);
            float temp2 = 0;
            for (int j = 0; j < mul_final_feature[0][0].length; j++) {
                temp2 += (mul_final_feature[i][1][j] - final_feature[1][j]) * (mul_final_feature[i][1][j] - final_feature[1][j]);
            }
            temp2 = (float) Math.sqrt(temp2);
//            score += (temp1 + temp2) / 2;
            score += temp1;
        }
        score = score / mul_final_feature.length;
        if (score < 0.5) {
            predittag = 1;
        }
        Log.e(">>>", "score:" + score);
        Log.e(">>>", "predittag:" + predittag);
        return predittag;
    }

    public int behavior_predit(float[] final_feature) {
        int predittag = 0;
        float score = 0;
        for (int i = 0; i < single_final_feature.length; i++) {
            float temp = 0;
            for (int j = 0; j < single_final_feature[0].length; j++) {
                temp += (single_final_feature[i][j] - final_feature[j]) * (single_final_feature[i][j] - final_feature[j]);
            }
            temp = (float) Math.sqrt(temp);
            score += temp;
        }
        score = score / single_final_feature.length;
        if (score < 0.5) {
            predittag = 1;
        }
        Log.e(">>>", "score:" + score);
        Log.e(">>>", "predittag:" + predittag);
        return predittag;
    }



//    public static void inputStream2File(InputStream is, File file) throws IOException {
//        OutputStream os = null;
//        try {
//            os = new FileOutputStream(file);
//            int len = 0;
//            byte[] buffer = new byte[8192];
//
//            while ((len = is.read(buffer)) != -1) {
//                os.write(buffer, 0, len);
//            }
//        } finally {
//            os.close();
//            is.close();
//        }
//    }
}
