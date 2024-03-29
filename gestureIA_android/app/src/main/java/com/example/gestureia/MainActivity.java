package com.example.gestureia;

import android.Manifest;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;

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

import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;


public class MainActivity extends WearableActivity {

    private Button button_register;
    private Button button_listen_start;
    private Button button_listen_stop;
    private Spinner gesture_spinner;

    private Button button_add;
    private Button button_final;

    private TextView gesturecount;
    private TextView tips;

    //实验室记录上传数据的服务器地址
    private String RequestURL = "http://10.28.194.222:8888/IA";

    //录入的手势条数
    private int count = 0;
    //录入状态标记，0是不可录入，1是可录入
    private int uploadtag = 1;
    private Timer timer = null;

    //传感器控制、电源控制、认证模型导入
    private Sensorcontrol sensors = null;
    private Energycontrol energycontrolr = null;
    private Baedmodel basedmodel = null;

    private long currenttime = 0;
    private long ensuretime = 0;

    private ArrayList<float[][]> mul_features_record = new ArrayList<float[][]>();
    private ArrayList<float[]> single_features_record = new ArrayList<float[]>();
    private int service_flag = 0;

    private double[] featureset = null;
    private double[][] madata = null;
    private Message msg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setAmbientEnabled();
//        引导开始传感器访问权限
        permissionrequest();

//        加载下拉列表
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.spinnervalue, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

//        注册按钮
        gesture_spinner = (Spinner) findViewById(R.id.gestureSpinner);
        gesture_spinner.setAdapter(adapter);
        button_register = (Button) findViewById(R.id.register);
        button_listen_start = (Button) findViewById(R.id.listen_start);
        button_listen_stop = (Button) findViewById(R.id.listen_stop);

//        后台service监听，判断是否存在手势
        button_listen_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.e(">>.", "service start！");
                //获取当前注册样本的特征
                if (0 == service_flag) {
                    basedmodel = new Baedmodel();
                    basedmodel.readmulfeature(getApplicationContext());
//                    basedmodel.readfeature(getApplicationContext());
                    if (basedmodel.single_final_feature == null && basedmodel.mul_final_feature == null) {
                        Toast.makeText(getApplicationContext(), "请选择先注册手势！", Toast.LENGTH_LONG).show();
                        Log.e(">>.", "请选择先注册手势！");
                        basedmodel = null;
                    } else {
                        basedmodel = null;
                        Toast.makeText(getApplicationContext(), "手势检测服务已开始！", Toast.LENGTH_LONG).show();
                        startService(new Intent(getBaseContext(), Behaviorlisten.class));
                        service_flag = 1;

//                      获取当前设备已安装的包名
//                    List<PackageInfo> packages = getPackageManager().getInstalledPackages(0);
//                    for(int i=0;i<packages.size();i++){
//                        Log.e(">>>","packet name:"+packages.get(i).packageName);
//                    }
                        //启动智能手表的支付宝
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
                }
                System.gc();
            }
        });
        //暂停服务
        button_listen_stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (1 == service_flag) {
                    Log.e(">>.", "service stop！");
                    Toast.makeText(getApplicationContext(), "手势检测服务已结束！", Toast.LENGTH_LONG).show();
                    stopService(new Intent(getBaseContext(), Behaviorlisten.class));
                    service_flag = 0;
                    System.gc();
                }
            }
        });

        //注册样本
        button_register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                final String gesture_item = gesture_spinner.getSelectedItem().toString();
                int lens = gesture_item.length();
                if (lens > 3) {
//                if (1 == 0) {
                    Toast.makeText(getApplicationContext(), "请选择要登记的手势！", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(getApplicationContext(), "系统初始化，请稍后！", Toast.LENGTH_LONG).show();
                    Log.e(">>>", "初始化，请稍等！");
                    uploadtag = 0;
                    setcount();
                    setContentView(R.layout.gesture_record);
                    ensuretime = System.currentTimeMillis();
                    button_final = (Button) findViewById(R.id.stopcount);
                    button_add = (Button) findViewById(R.id.addcount);
                    gesturecount = (TextView) findViewById(R.id.count);
                    tips = (TextView) findViewById(R.id.tips);

                    tips.setText("初始化中，请稍等！");


                    sensors = new Sensorcontrol();
                    energycontrolr = new Energycontrol();
                    timer = new Timer();
                    basedmodel = new Baedmodel();

                    energycontrolr.energyopen(getApplicationContext());
                    sensors.StartSensorListening(getApplicationContext());
                    basedmodel.readmodelpara(getApplicationContext());


                    //样本注册完成
                    button_final.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            Toast.makeText(getApplicationContext(), "已登记手势样本" + getcount() + "条", Toast.LENGTH_LONG).show();
                            energycontrolr.energyclose();
                            sensors.StopSensorListening();
                            timer.cancel();
                            sensors = null;
                            energycontrolr = null;
                            uploadtag = 0;
//                            timer = null;
                            //将最终向量保存到文件中
                            if (mul_features_record.size() > 0 || single_features_record.size() > 0) {
                                Log.e(">>>", "确认手势样本" + mul_features_record.size() + "条");
                                basedmodel.writemulfeature(getApplicationContext(), mul_features_record);

//                                Log.e(">>>", "确认手势样本" + single_features_record.size() + "条");
//                                basedmodel.writefeature(getApplicationContext(), single_features_record,"basedfeature.csv");

                                basedmodel = null;
                                System.gc();
                            } else {
                                Log.e(">>>", "当前无手势特征录入文件");
                            }
                            Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                            startActivity(intent);
                        }
                    });
                    //注册单个手势片段
                    button_add.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            if (uploadtag == 1) {
//                                fossil的是100hz，tic的是200hz
                                uploadtag = 3;
                                PPG ppgs = sensors.getnewppgseg(1600);
                                Motion motions = sensors.getnewmotionseg(800);
                                //计算当前片段通过网络后的向量

                                Featurecontrol featurecontrols = new Featurecontrol();
                                featureset = featurecontrols.build_feature(ppgs, motions);
//                                madata = featurecontrols.build_madata(ppgs, motions);
                                segmentupload(RequestURL, motions, ppgs, 10000, "s_segment", gesture_item);

                                if (featureset != null) {
                                    msg = new Message();
                                    msg.what = 0;
                                    handler.sendMessage(msg);

                                } else {
                                    msg = new Message();
                                    msg.what = 1;
                                    handler.sendMessage(msg);
                                }
                                featurecontrols = null;
                                ppgs = null;
                                motions = null;
                                if (featureset != null) {
                                    mul_features_record.add(basedmodel.featureprocess(featureset));
                                }
//                                if (madata != null) {
//                                    single_features_record.add(basedmodel.dataprocess(madata));
//                                }
                                System.gc();
                            } else {
                                msg = new Message();
                                msg.what = 7;
                                handler.sendMessage(msg);
                            }
                        }
                    });

                    //定时更新页面片段
                    timer.schedule(new TimerTask() {
                        @Override
                        public void run() {
                            Log.e(">>>", "ppg.newindex:" + sensors.getppgsize() + " acc.newindex:" + sensors.getaccsize() + " gyr.newindex:" + sensors.getgyrsize());
                            System.out.println("TimerTask");
                            currenttime = System.currentTimeMillis();
//                            Log.e(">>>>",""+(currenttime - starttime)+","+(currenttime - ensuretime));
                            if (((currenttime - ensuretime) > 9000) && (uploadtag == 2)) {
                                //错误录入手势后的等待时间过去了
                                msg = new Message();
                                msg.what = 6;
                                handler.sendMessage(msg);
                                uploadtag = 1;
                            }
                            if (((currenttime - ensuretime) > 7000) && (uploadtag == 0)) {
                                //正确录入手势后的等待时间过去了
                                msg = new Message();
                                msg.what = 6;
                                handler.sendMessage(msg);
                                uploadtag = 1;
                            }
                            System.gc();
                        }
                    }, 10, 1000);
                }
            }
        });
    }

    private Handler handler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == 0) {
                countupdate();
                int i = getcount();
                Log.e(">>>", "segment：" + i);
                //手势可以被提取出来
                ensuretime = System.currentTimeMillis();
                uploadtag = 0;
                gesturecount.setText(String.valueOf(i));
                tips.setText("录入中，请稍等！");
                Log.e(">>>", "录入中，请稍等！");
            }
            if (msg.what == 1) {
                ensuretime = System.currentTimeMillis();
                uploadtag = 2;
                Toast.makeText(getApplicationContext(), "手势提取失败！", Toast.LENGTH_LONG).show();
                Log.e(">>>", "手势提取失败！");
                tips.setText("请稍等！");
            }
            if (msg.what == 2) {
                Toast.makeText(getApplicationContext(), "上传成功！", Toast.LENGTH_LONG).show();
            }
            if (msg.what == 3) {
                Toast.makeText(getApplicationContext(), "上传失败！", Toast.LENGTH_LONG).show();
            }
            if (msg.what == 6) {
                tips.setText("请录入手势！");
                Log.e(">>>", "请录入手势！");
            }
            if (msg.what == 7) {
                Toast.makeText(getApplicationContext(), "请稍等！", Toast.LENGTH_LONG).show();
            }
        }
    };


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
            //若用户未开启权限，则引导用户开启权限
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
//
//    private void readbasedfeature2() {
//        try {
//            ArrayList<String> temp_feature = new ArrayList<String>();
//            InputStream parameterinput = getAssets().open("featuredataset/1.csv");
//            BufferedReader reader = new BufferedReader(new InputStreamReader(parameterinput));
//            String line;
//            while ((line = reader.readLine()) != null) {
//                temp_feature.add(line);
//            }
//            reader.close();
//            //提取128位最终的向量
//            String[][] str_feature = new String[temp_feature.size()][];
//            final_feature = new Double[temp_feature.size()][128];
//            for (int i = 0; i < temp_feature.size(); i++) {
//                str_feature[i] = temp_feature.get(i).split(",");
//            }
//            for (int i = 0; i < temp_feature.size(); i++) {
//                for (int j = 0; j < 128; j++) {
//                    final_feature[i][j] = Double.parseDouble(str_feature[i][j]);
//                }
//            }
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

    private void testtf() {
        //获取特征参数

        long startime = System.currentTimeMillis();
        Log.e(">>>", "测试开始时间：" + startime);

//
//            ArrayList<double[]> featureset = new ArrayList<double[]>();
//
//            InputStream featurefile = getAssets().open("oridataset/1.csv");
//            PPG ppg1 = filecontrols.ppgreadin(featurefile);
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


//    public void sample_predict(final Double[][] final_feature, final Double[][] inform_feature) {
//        //Runs inference in background thread
//        new AsyncTask<Integer, Integer, Integer>() {
//
//            @Override
//            protected Integer doInBackground(Integer... params) {
//                int datalen = inform_feature.length;
//                float[][] datax = new float[datalen][30];
//                for (int i = 0; i < datalen; i++) {
//                    for (int j = 0; j < 30; j++) {
//                        datax[i][j] = (float) (double) inform_feature[i][j];
//                    }
//                }
//                Log.d(">>>", " model load success");
//                float[][] outPutsx = new float[datalen][128];//结果分类
//                tflite.run(datax, outPutsx);
//                float[] score = new float[datalen];
//                for (int i = 0; i < datalen; i++) {
//                    float temp = 0;
//                    for (int j = 0; j < final_feature.length; j++) {
//                        for (int k = 0; k < 128; k++) {
//                            temp = temp + (outPutsx[i][k] - (float) (double) final_feature[j][k]) * (outPutsx[i][k] - (float) (double) final_feature[j][k]);
//                        }
//                    }
//                    temp = temp / 5;
//
//                    score[i] = (float) Math.sqrt(temp);
//                }
//                for (int i = 0; i < datalen; i++) {
//                    Log.e(">>>", "score[i]:" + score[i] + "," + i);
//                }
//                long stoptime = System.currentTimeMillis();
//                Log.e(">>>", "测试结束时间：" + stoptime);
//                return 0;
//            }
//
//        }.execute(0);
//
//    }
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
//                float[][] outPutsx = new float[datalen][128];//结果分类
//                float[][] outPutsy = new float[datalen][128];//结果分类
//
//                tflite.run(datax, outPutsx);
//                tflite.run(datay, outPutsy);
//                float[] score = new float[datalen];
//                for (int i = 0; i < datalen; i++) {
//                    float temp = 0;
//                    for (int j = 0; j < 128; j++) {
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
    public void segmentupload(final String RequestURL, final Motion motions, final PPG ppgs, final int TIME_OUT, final String tag, final String item) {
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

                    data.put("username", "tempuser7");
                    data.put("sensor", tag);
                    data.put("gesture_item", item);

                    Log.e("cat", ">>>" + data.toString());
                    sb.append("Content-Disposition: form-data; name=\"data\";filename=\"" + data.toString() + "\"" + LINE_END);
                    sb.append("Content-Type: application/octet-stream; charset=" + CHARSET + LINE_END);
                    sb.append(LINE_END);
                    dos.write(sb.toString().getBytes());
                    sb = null;
                    StringBuffer sd = new StringBuffer();
                    for (int i = 0; i < motions.accx.length; i++) {
                        sd.append(0).append(",")
                                .append(motions.accx[i]).append(",")
                                .append(motions.accy[i]).append(",")
                                .append(motions.accz[i]).append(",")
                                .append(motions.acctimestamps[i]).append(",")
                                .append("\n");
                    }
                    for (int i = 0; i < motions.gyrx.length; i++) {
                        sd.append(1).append(",")
                                .append(motions.gyrx[i]).append(",")
                                .append(motions.gyry[i]).append(",")
                                .append(motions.gyrz[i]).append(",")
                                .append(motions.gyrtimestamps[i]).append(",")
                                .append("\n");
                    }

                    for (int i = 0; i < ppgs.x.length; i++) {
                        sd.append(2).append(",")
                                .append(ppgs.x[i]).append(",")
                                .append(ppgs.y[i]).append(",")
                                .append(ppgs.timestamps[i]).append(",")
                                .append("\n");
                    }
//                    Log.e(">>>", "" + accx.size() + " " + gyrx.size() + " " + orix.size() + " " + magx.size());
                    dos.write(sd.toString().getBytes());
                    dos.write(LINE_END.getBytes());
                    byte[] end_data = (PREFIX + BOUNDARY + PREFIX + LINE_END).getBytes();
                    dos.write(end_data);
                    dos.flush();
                    dos = null;
                    sd = null;

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
