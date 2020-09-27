package com.example.gestureia;

import android.Manifest;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;

import org.tensorflow.lite.Interpreter;

import android.content.res.AssetFileDescriptor;

import java.io.FileInputStream;

import static java.lang.Math.abs;

public class MainActivity extends WearableActivity {

    private Button button_register;
    private Button button_listen;
    private Spinner gesture_spinner;

    private Button button_add;
    private Button button_final;

    private TextView gesturecount;
    private TextView tips;

    //实验室记录上传数据的服务器地址
//    private String RequestURL = "http://192.168.1.101:8888/IA";

    //录入的手势条数
    private int count = 0;
    //录入状态标记，0是不可录入，1是可录入
    private int uploadtag = 1;
    private Timer timer = new Timer();

    private Sensorcontrol sensors = new Sensorcontrol();
    private IAtool iatools = new IAtool();
    private Normal_tool nortools = new Normal_tool();
    private Filecontrol filecontrols = new Filecontrol();
    private Featurecontrol featurecontrols = new Featurecontrol();
    private siamese_model siamese = new siamese_model();
    private long starttime = 0;
    private long currenttime = 0;
    private long ensuretime = 0;

    private Integer[] sort1 = null;
    //          private   Integer[] sort2 = nortools.strarraytointarray(strsort2);
    private Double[] scale_mean = null;
    private Double[] scale_scale = null;
    private Double[][] final_feature = null;
    private Interpreter tflite = null;

    private ArrayList<Ppg> ppg_record = new ArrayList<Ppg>();
    private ArrayList<Motion> motion_record = new ArrayList<Motion>();
    private ArrayList<float[]> features_record = new ArrayList<float[]>();


    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setAmbientEnabled();
        permissionrequest();
        readmodelpara();
        readbasedfeature();
//
//        testtf();
        //加载下拉列表
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.spinnervalue, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gesture_spinner = (Spinner) findViewById(R.id.gestureSpinner);
        gesture_spinner.setAdapter(adapter);


        button_register = (Button) findViewById(R.id.register);
        button_listen = (Button) findViewById(R.id.listen);

//        后台监听
        button_listen.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (final_feature == null) {
                    Toast.makeText(getApplicationContext(), "请选择先注册手势！", Toast.LENGTH_LONG).show();
                } else {
                    startService(new Intent(getBaseContext(), behaviorlisten.class));
                    //获取当前设备已安装的包名
//                    List<PackageInfo> packages = getPackageManager().getInstalledPackages(0);
//                    for(int i=0;i<packages.size();i++){
//                        Log.e(">>>","packet name:"+packages.get(i).packageName);
//                    }
                    //启动支付宝
//                    PackageManager pm = getApplication().getPackageManager();
//                    try {
//                        pm.getPackageInfo("com.eg.android.AlipayGphone", PackageManager.GET_ACTIVITIES);
//                        Intent intent = pm.getLaunchIntentForPackage("com.eg.android.AlipayGphone");
//                        startActivity(intent);
//                    } catch (PackageManager.NameNotFoundException e) {
//                        e.printStackTrace();
//                        Log.e(">>>","当前设备不存在支付宝应用，请前往应用市场下载。");
//                    }


                }
//                stopService(new Intent(getBaseContext(), sensorlisten.class));



            }
        });
        //注册样本
        button_register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final String gesture_item = gesture_spinner.getSelectedItem().toString();
                int lens = gesture_item.length();
                if (lens > 3) {
                    Toast.makeText(getApplicationContext(), "请选择要登记的手势！", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(getApplicationContext(), "系统初始化，请稍后！", Toast.LENGTH_LONG).show();
                    setContentView(R.layout.gesture_record);

                    iatools.energyopen(getApplicationContext());
                    sensors.dataclear();
                    setcount();

                    starttime = System.currentTimeMillis();
                    ensuretime = System.currentTimeMillis();

                    sensors.StartSensorListening(getApplicationContext());

                    button_final = (Button) findViewById(R.id.stopcount);
                    button_add = (Button) findViewById(R.id.addcount);
                    gesturecount = (TextView) findViewById(R.id.count);
                    tips = (TextView) findViewById(R.id.tips);


                    button_final.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            Toast.makeText(getApplicationContext(), "已登记手势样本" + getcount() + "条", Toast.LENGTH_LONG).show();
                            timer.cancel();
                            sensors.StopSensorListening();
                            iatools.energyclose();
                            //将最终向量保存到文件中
                            for (int i = 0; i < final_feature.length; i++) {
                                float[] temp = new float[final_feature[i].length];
                                for (int j = 0; j < final_feature[i].length; j++) {
                                    temp[j] = (float) (double) final_feature[i][j];
                                }
                                features_record.add(temp);
                            }
                            filecontrols.basedfeaturewrite(getApplicationContext(), features_record);
                            Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                            startActivity(intent);

//                            实验室中，上传数据到服务器
//                            ppg rawppgs = sensors.getnewppgseg();
//                            motion motions = sensors.getnewmotionseg();
//                            //上传完整的长片段
//                            if (getcount() > 0 || rawppgs.x.size() < 100000) {
//                                segmentupload(RequestURL, motions, rawppgs, 300000, "l_segment", gesture_item);
////                                files.segmentupload_new(RequestURL,getApplicationContext(),motions, rawppgs, 300000, "l_segment", gesture_item);
//                            }
                        }
                    });

                    button_add.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            if (uploadtag == 1) {
                                ensuretime = System.currentTimeMillis();
                                countupdate();
                                int i = getcount();
                                Log.e(">>>", "segment：" + i);
                                gesturecount.setText(String.valueOf(i));
                                Ppg rawppgs = sensors.getnewppgseg(1800);
                                Motion motions = sensors.getnewmotionseg(900);
                                //保存当前选择的数据段
                                ppg_record.add(rawppgs);
                                motion_record.add(motions);

                                Message msg = new Message();
                                msg.what = 1;
                                handler.sendMessage(msg);
                                //计算当前片段通过网络后的向量
                                ArrayList<Double> temp = featurecontrols.sample_feature(rawppgs);
                                Double[] inform_feature = new Double[30];
                                for (i = 0; i < 30; i++) {
                                    inform_feature[i] = temp.get(sort1[i]);
                                }
                                inform_feature = iatools.featurestd(inform_feature, scale_mean, scale_scale);
                                features_record.add(siamese.sample_feature(tflite, inform_feature));


                                //实验室中上传人工确认后的短片段
//                                ppg rawppgs = sensors.getnewppgseg(1800);
//                                motion motions = sensors.getnewmotionseg(900);
//                                segmentupload(RequestURL, motions, rawppgs, 10000, "s_segment", gesture_item);

                            } else {
                                Toast.makeText(getApplicationContext(), "请稍等！", Toast.LENGTH_LONG).show();
                            }
                        }
                    });

                    timer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            System.out.println("TimerTask");
                            currenttime = System.currentTimeMillis();
//                            Log.e(">>>>",""+(currenttime - starttime)+","+(currenttime - ensuretime));
                            if (((currenttime - starttime) < 7000) && (uploadtag != 0)) {
                                //开始录入时的等待
                                Message msg = new Message();
                                msg.what = 4;
                                handler.sendMessage(msg);
                                uploadtag = 0;
                            } else {
                                if (((currenttime - ensuretime) < 7000) && (uploadtag != 0)) {
                                    //录入一条手势时的等待
                                    Message msg = new Message();
                                    msg.what = 5;
                                    handler.sendMessage(msg);
                                    uploadtag = 0;
                                }
                                if (((currenttime - ensuretime) > 7000) && (uploadtag == 0)) {
                                    //录入手势后的等待时间过去了
                                    Message msg = new Message();
                                    msg.what = 6;
                                    handler.sendMessage(msg);
                                    uploadtag = 1;
                                }
                            }
//                            if(1==uploadtag){
//                                ppg ppgs = sensors.getnewppgseg(1400);
//                                double[] temp=nortools.arraytomatrix(ppgs.x);
//                                double[] orippgx = nortools.meanfilt(nortools.array_dataselect(temp,temp.length-1000,1000), 20);
//                                int tag = ma.coarse_grained_detect(orippgx);
//                                System.out.println("tag:" + tag);
//                                if(1==tag){
//                                    ensuretime = System.currentTimeMillis();
//
//                                    Message msg = new Message();
//                                    msg.what = 1;
//                                    handler.sendMessage(msg);
//                                    countupdate();
//
//
//
//                                }
//                            }

                        }
                    }, 10, 1000);
                }
            }
        });
    }

    private Handler handler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == 1) {
                int i = getcount();
                Log.e(">>>", "segment：" + i);
                gesturecount.setText(String.valueOf(i));
            }
            if (msg.what == 2) {
                Toast.makeText(getApplicationContext(), "上传成功！", Toast.LENGTH_LONG).show();
            }
            if (msg.what == 3) {
                Toast.makeText(getApplicationContext(), "上传失败！", Toast.LENGTH_LONG).show();
            }
            if (msg.what == 4) {
                tips.setText("初始化中，请稍等！");
                Log.e(">>>", "初始化，请稍等！");
            }
            if (msg.what == 5) {
                tips.setText("录入中，请稍等！");
                Log.e(">>>", "录入中，请稍等！");
            }
            if (msg.what == 6) {
                tips.setText("请录入手势！");
                Log.e(">>>", "请录入手势！");
            }
        }
    };


    private void readmodelpara() {
        try {
            tflite = new Interpreter(loadModelFile("based_model"));
            InputStream parameterinput = getAssets().open("stdpropara.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(parameterinput));
            String temp_sort1 = reader.readLine();
//            String temp_sort2 = reader.readLine();
            String temp_scale_mean = reader.readLine();
            String temp_scale_scale = reader.readLine();
            reader.close();
            parameterinput.close();

            String[] str_sort1 = temp_sort1.replace("[", "").replace("]", "").replace(" ", "").split(",");
//            String[] strsort2 = temp_sort2.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_mean = temp_scale_mean.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_scale = temp_scale_scale.replace("[", "").replace("]", "").replace(" ", "").split(",");

            sort1 = nortools.strarraytointarray(str_sort1);
//            Integer[] sort2 = nortools.strarraytointarray(strsort2);
            scale_mean = nortools.strarraytodoublearray(str_scale_mean);
            scale_scale = nortools.strarraytodoublearray(str_scale_scale);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(String model) throws IOException {
        Log.e(">>>", model + ".tflite");
        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private void readbasedfeature() {
        try {
            ArrayList<String> temp_feature = new ArrayList<String>();
            String fileName = getExternalFilesDir("").getAbsolutePath() + "basedfeature.csv";//文件存储路径
            Log.e(">>>", "basedfeature filename:" + fileName);
            File file = new File(fileName);
            if (file.exists()) {
                BufferedReader reader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = reader.readLine()) != null) {
                    temp_feature.add(line);
                }
                reader.close();
                //提取256位最终的向量
                int listnum = 0;
                if (temp_feature.size() > 5) {
                    listnum = 5;
                } else {
                    listnum = temp_feature.size();
                }
                String[][] str_feature = new String[listnum][];
                final_feature = new Double[listnum][256];
                for (int i = 0; i < listnum; i++) {
                    str_feature[i] = temp_feature.get(i).split(",");
                }
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < 256; j++) {
                        final_feature[i][j] = Double.parseDouble(str_feature[i][j]);
                    }
                }

            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private int getcount() {
        return count;
    }

    private void countupdate() {
        count = count + 1;
    }

    private void setcount() {
        count = 0;
    }

    private void permissionrequest() {
        if (!hasPermission()) {
            //若用户未开启权限，则引导用户开启“Apps with usage access”权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.BODY_SENSORS}, 0);
        }
    }

    private boolean hasPermission() {
        AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        int sensor = ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS);
        boolean map = false;
        if (sensor == PackageManager.PERMISSION_GRANTED)
            map = true;
        return map;
    }

    private void readbasedfeature2() {
        try {
            ArrayList<String> temp_feature = new ArrayList<String>();
            InputStream parameterinput = getAssets().open("featuredataset/1.csv");
            BufferedReader reader = new BufferedReader(new InputStreamReader(parameterinput));
            String line;
            while ((line = reader.readLine()) != null) {
                temp_feature.add(line);
            }
            reader.close();
            //提取256位最终的向量
            String[][] str_feature = new String[temp_feature.size()][];
            final_feature = new Double[temp_feature.size()][256];
            for (int i = 0; i < temp_feature.size(); i++) {
                str_feature[i] = temp_feature.get(i).split(",");
            }
            for (int i = 0; i < temp_feature.size(); i++) {
                for (int j = 0; j < 256; j++) {
                    final_feature[i][j] = Double.parseDouble(str_feature[i][j]);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void testtf() {
        //获取特征参数

        long startime = System.currentTimeMillis();
        Log.e(">>>", "测试开始时间：" + startime);

//
//            ArrayList<double[]> featureset = new ArrayList<double[]>();
//
//            InputStream featurefile = getAssets().open("oridataset/1.csv");
//            Ppg ppg1 = filecontrols.ppgreadin(featurefile);
//            Featurecontrol sampleFeaturecontrol = single_feature(ppg1);
//            featureset.add(sampleFeaturecontrol);
//
//            featurefile = getAssets().open("oridataset/2.csv");
//            ppg1 = filecontrols.ppgreadin(featurefile);
//            sampleFeaturecontrol = single_feature(ppg1);
//            featureset.add(sampleFeaturecontrol);
//
//            featurefile = getAssets().open("oridataset/3.csv");
//            ppg1 = filecontrols.ppgreadin(featurefile);
//            sampleFeaturecontrol = single_feature(ppg1);
//            featureset.add(sampleFeaturecontrol);
//
//            featurefile = getAssets().open("oridataset/4.csv");
//            ppg1 = filecontrols.ppgreadin(featurefile);
//            sampleFeaturecontrol = single_feature(ppg1);
//            featureset.add(sampleFeaturecontrol);
//
//            featurefile = getAssets().open("oridataset/5.csv");
//            ppg1 = filecontrols.ppgreadin(featurefile);
//            sampleFeaturecontrol = single_feature(ppg1);
//            featureset.add(sampleFeaturecontrol);

        long innerime = System.currentTimeMillis();
        Log.e(">>>", "特征提取结束时间：" + innerime);

        //
//            Log.e(">>>","sort1:"+sort1[0]);
//            Log.e(">>>","sort2:"+sort2[0]);
//            Log.e(">>>","scale_mean:"+scale_mean[0]);
//            Log.e(">>>","scale_scale:"+scale_scale[0]);
//            Double[][] ppg_feature = new Double[featureset.size()][84];
//            for (int i = 0; i < featureset.size(); i++) {
//                double[] temp = nortools.arraytomatrix(featureset.get(i).features);
//                for (int j = 0; j < 84; j++) {
//                    ppg_feature[i][j] = temp[j];
//                }
//            }
//            Double[][] inform_feature = new Double[ppg_feature.length][30];
//            for (int i = 0; i < featureset.size(); i++) {
//                for (int j = 0; j < 30; j++) {
//                    inform_feature[i][j] = ppg_feature[i][sort1[j]];
//                }
//            }

//            inform_feature = iatools.featurestd(inform_feature, scale_mean, scale_scale);
//            innerime = System.currentTimeMillis();
//            Log.e(">>>", "数据处理结束时间：" + innerime);

//            featurefile = getAssets().open("dataset/2.csv");
//            reader = new BufferedReader(new InputStreamReader(featurefile));
//            while ((line = reader.readLine()) != null) {
//                temp_feature.add(line);
//                target.add(1);
//            }
//            reader.close();
//            featurefile.close();

        //将文件中的特征转为30位向量
//            String[][] str_feature = new String[temp_feature.size()][];
//            for (int i = 0; i < temp_feature.size(); i++) {
//                str_feature[i] = temp_feature.get(i).split(",");
//            }
//            Double[][] ppg_feature = new Double[temp_feature.size()][84];
//            for (int i = 0; i < temp_feature.size(); i++) {
//                Double[] temp = nortools.strarraytodoublearray(str_feature[i]);
//                for (int j = 0; j < 84; j++) {
//                    ppg_feature[i][j] = temp[j];
//                }
//            }
//            Double[][] inform_feature = new Double[ppg_feature.length][30];
//            for (int i = 0; i < temp_feature.size(); i++) {
//                for (int j = 0; j < 30; j++) {
//                    inform_feature[i][j] = ppg_feature[i][sort1[j]];
//                }
//            }
//            Log.e(">>>", "feature:" + inform_feature[0][0]);
//            ppg_feature = iatools.featurestd(inform_feature, scale_mean, scale_scale);
//            Log.e(">>>", "feature:" + ppg_feature[0][0]);


//            Integer[] targets = new Integer[target.size()];
//            for (int i = 0; i < target.size(); i++) {
//                targets[i] = target.get(i);
//            }
//            datapair pairs = iatools.create_pairs(ppg_feature, targets, 2);
//            predict(pairs);


//            sample_predict(final_feature, inform_feature);

    }


    public void sample_predict(final Double[][] final_feature, final Double[][] inform_feature) {
        //Runs inference in background thread
        new AsyncTask<Integer, Integer, Integer>() {

            @Override
            protected Integer doInBackground(Integer... params) {
                int datalen = inform_feature.length;
                float[][] datax = new float[datalen][30];
                for (int i = 0; i < datalen; i++) {
                    for (int j = 0; j < 30; j++) {
                        datax[i][j] = (float) (double) inform_feature[i][j];
                    }
                }
                Log.d(">>>", " model load success");
                float[][] outPutsx = new float[datalen][256];//结果分类
                tflite.run(datax, outPutsx);
                float[] score = new float[datalen];
                for (int i = 0; i < datalen; i++) {
                    float temp = 0;
                    for (int j = 0; j < final_feature.length; j++) {
                        for (int k = 0; k < 256; k++) {
                            temp = temp + (outPutsx[i][k] - (float) (double) final_feature[j][k]) * (outPutsx[i][k] - (float) (double) final_feature[j][k]);
                        }
                    }
                    temp = temp / 5;

                    score[i] = (float) Math.sqrt(temp);
                }
                for (int i = 0; i < datalen; i++) {
                    Log.e(">>>", "score[i]:" + score[i] + "," + i);
                }
                long stoptime = System.currentTimeMillis();
                Log.e(">>>", "测试结束时间：" + stoptime);
                return 0;
            }

        }.execute(0);

    }
//
//    public void pair_predict(final Datapair pairs) {
//        //Runs inference in background thread
//        new AsyncTask<Integer, Integer, Integer>() {
//
//            @Override
//            protected Integer doInBackground(Integer... params) {
//
//
//                int datalen = pairs.x.size();
//                double[][] temp_datax = new double[datalen][30];
//                double[][] temp_datay = new double[datalen][30];
//                Integer[] label = new Integer[datalen];
//                for (int i = 0; i < datalen; i++) {
//                    for (int j = 0; j < 30; j++) {
//                        temp_datax[i][j] = pairs.x.get(i)[j];
//                        temp_datay[i][j] = pairs.y.get(i)[j];
//                        label[i] = pairs.label.get(i);
//                    }
//                }
//
//                float[][] datax = new float[datalen][30];
//                float[][] datay = new float[datalen][30];
//                for (int i = 0; i < datalen; i++) {
//                    for (int j = 0; j < 30; j++) {
//                        datax[i][j] = (float) temp_datax[i][j];
//                        datay[i][j] = (float) temp_datay[i][j];
//                    }
//                }
//
//                Boolean load_result;
//                Log.d(">>>", " model load success");
//                float[][] outPutsx = new float[datalen][256];//结果分类
//                float[][] outPutsy = new float[datalen][256];//结果分类
//
//                tflite.run(datax, outPutsx);
//                tflite.run(datay, outPutsy);
//                float[] score = new float[datalen];
//                for (int i = 0; i < datalen; i++) {
//                    float temp = 0;
//                    for (int j = 0; j < 256; j++) {
//                        temp = temp + (outPutsx[i][j] - outPutsy[i][j]) * (outPutsx[i][j] - outPutsy[i][j]);
//                    }
//                    score[i] = (float) Math.sqrt(temp);
//                }
//                for (int i = 0; i < datalen; i++) {
//                    Log.e(">>>", "score[i]:" + score[i] + "," + i);
//                }
//                double t = 0.01;
//                while (t < 3) {
//                    int tp = 0;
//                    int tn = 0;
//                    int fp = 0;
//                    int fn = 0;
//                    for (int j = 0; j < datalen; j++) {
//                        if (score[j] < t) {
//                            if (1 == label[j]) {
//                                tp = tp + 1;
//                            } else {
//                                fp = fp + 1;
//                            }
//                        } else {
//                            if (1 == label[j]) {
//                                fn = fn + 1;
//                            } else {
//                                tn = tn + 1;
//                            }
//                        }
//                    }
////                        Log.e(">>>","tp:"+tp+",fp:"+fp+",frr:"+frr);
//
//                    double accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
//                    double far = (double) (fp) / (fp + tn);
//                    double frr = (double) (fn) / (fn + tp);
//                    if ((frr < far) || (abs(frr - far) < 0.02)) {
//                        Log.e(">>>", "accuracy:" + accuracy + ",far:" + far + ",frr:" + frr);
//                        break;
//                    }
//                    t = t + 0.01;
//                }
//
//                return 0;
//            }
//
//        }.execute(0);
//
//    }

    //将实验数据上传至服务器
    public void segmentupload(final String RequestURL, final Motion motions, final Ppg ppgs, final int TIME_OUT, final String tag, final String item) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                final String CHARSET = "utf-8"; //设置编码
                BufferedReader reader = null;
                String result = null;
                String BOUNDARY = UUID.randomUUID().toString();  //边界标识   随机生成
                String PREFIX = "--", LINE_END = "\r\n";
                String CONTENT_TYPE = "multipart/form-data";   //内容类型
                try {
                    URL url = new URL(RequestURL);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setReadTimeout(TIME_OUT);
                    conn.setConnectTimeout(TIME_OUT);
                    conn.setDoInput(true);  //允许输入流
                    conn.setDoOutput(true); //允许输出流
                    conn.setUseCaches(false);  //不允许使用缓存
                    conn.setRequestMethod("POST");  //请求方式
                    conn.setRequestProperty("Charset", CHARSET);  //设置编码
                    conn.setRequestProperty("connection", "keep-alive");
                    conn.setRequestProperty("Content-Type", CONTENT_TYPE + ";boundary=" + BOUNDARY);
                    DataOutputStream dos = new DataOutputStream(conn.getOutputStream());
                    StringBuffer sb = new StringBuffer();
                    sb.append(PREFIX);
                    sb.append(BOUNDARY);
                    sb.append(LINE_END);
                    JSONObject data = new JSONObject();

                    data.put("username", "tempuser");
                    data.put("sensor", tag);
                    data.put("gesture_item", item);

                    Log.e("cat", ">>>" + data.toString());
                    sb.append("Content-Disposition: form-data; name=\"data\";filename=\"" + data.toString() + "\"" + LINE_END);
                    sb.append("Content-Type: application/octet-stream; charset=" + CHARSET + LINE_END);
                    sb.append(LINE_END);
                    dos.write(sb.toString().getBytes());
                    StringBuffer sd = new StringBuffer();
                    for (int i = 0; i < motions.accx.size(); i++) {
                        sd.append(0).append(",")
                                .append(motions.accx.get(i)).append(",")
                                .append(motions.accy.get(i)).append(",")
                                .append(motions.accz.get(i)).append(",")
                                .append(motions.acctimestamps.get(i)).append(",")
                                .append("\n");
                    }
                    for (int i = 0; i < motions.gyrx.size(); i++) {
                        sd.append(1).append(",")
                                .append(motions.gyrx.get(i)).append(",")
                                .append(motions.gyry.get(i)).append(",")
                                .append(motions.gyrz.get(i)).append(",")
                                .append(motions.gyrtimestamps.get(i)).append(",")
                                .append("\n");
                    }

                    for (int i = 0; i < ppgs.x.size(); i++) {
                        sd.append(2).append(",")
                                .append(ppgs.x.get(i)).append(",")
                                .append(ppgs.y.get(i)).append(",")
                                .append(ppgs.timestamps.get(i)).append(",")
                                .append("\n");
                    }
//                    Log.e(">>>", "" + accx.size() + " " + gyrx.size() + " " + orix.size() + " " + magx.size());
                    dos.write(sd.toString().getBytes());
                    dos.write(LINE_END.getBytes());
                    byte[] end_data = (PREFIX + BOUNDARY + PREFIX + LINE_END).getBytes();
                    dos.write(end_data);
                    dos.flush();
                    int res = conn.getResponseCode();
                    Log.e(">>>", "response code:" + res);
                    Log.e(">>>", "request success");
                    InputStream in = conn.getInputStream();
                    reader = new BufferedReader(new InputStreamReader(in));
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) response.append(line);
                    JSONObject json = new JSONObject(response.toString());
                    Log.e("cat", ">>>>" + json);
                    JSONArray jsonlogin = json.getJSONArray("updata");
                    if (jsonlogin.opt(0).toString().equals("success")) {
                        Log.e(">>>", "updata success");
                        if (tag.equals("s_segment")) {
                            Message msg = new Message();
                            msg.what = 2;
                            handler.sendMessage(msg);
                        }
                    }
                    if (jsonlogin.opt(0).toString().equals("false")) {
                        Log.e(">>>", "updata false");
                        Message msg = new Message();
                        msg.what = 3;
                        handler.sendMessage(msg);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    Message msg = new Message();
                    msg.what = 3;
                    handler.sendMessage(msg);
                }

            }
        }).start();
    }


}
