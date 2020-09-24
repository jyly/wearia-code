package com.example.gestureia;

import android.Manifest;
import android.app.Activity;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.Settings;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

//import static android.net.wifi.SupplicantState.COMPLETED;
//import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
//import java.io.InputStream;
import org.tensorflow.lite.Interpreter;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import static java.lang.Math.abs;

public class MainActivity extends WearableActivity {
//
//    static {
//        System.loadLibrary("tensorflow_inference");
//        Log.i("wumei", "load tensorflow_inference successfully");
//    }

//    private String MODEL_PATH = "ppg_model_weights.pb";

//    private TensorFlowInferenceInterface tf;

    private Button button_start;
    private Button button_final;
    private Button button_add;
    private Spinner gesture_spinner;
    private TextView gesturecount;
    private TextView tips;

    private Button button_listen;


    private String RequestURL = "http://192.168.1.101:8888/IA";
    //    private String RequestURL = "http://47.94.87.104:4092/IA";
    private int count = 0;
    private int uploadtag = 0;
    private Timer timer = new Timer();

    private sensorcontrol sensors = new sensorcontrol();
    private IAtool iatools = new IAtool();
    private normal_tool nortools = new normal_tool();
    private MAfind ma = new MAfind();

    private long starttime = 0;
    private long currenttime = 0;
    private long ensuretime = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setAmbientEnabled();
        permissionrequest();
//        testtf();

        button_start = (Button) findViewById(R.id.start);
        button_listen = (Button) findViewById(R.id.listen);
        gesture_spinner = (Spinner) findViewById(R.id.gestureSpinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.spinnervalue, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        gesture_spinner.setAdapter(adapter);
//        后台监听
        button_listen.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                startService(new Intent(getBaseContext(), sensorlisten.class));


            }
        });
        //注册样本
        button_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                stopService(new Intent(getBaseContext(), sensorlisten.class));

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
                            Toast.makeText(getApplicationContext(), "已收集手势数据" + getcount() + "条", Toast.LENGTH_LONG).show();
                            timer.cancel();
                            sensors.StopSensorListening();
                            iatools.energyclose();

                            ppg rawppgs = sensors.getnewppgseg();
                            motion motions = sensors.getnewmotionseg();
                            //上传完整的长片段
                            if (getcount() > 0 || rawppgs.x.size() < 100000) {
                                segmentupload(RequestURL, motions, rawppgs, 300000, "l_segment", gesture_item);
//                                files.segmentupload_new(RequestURL,getApplicationContext(),motions, rawppgs, 300000, "l_segment", gesture_item);
                            }
                            Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                            startActivity(intent);
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
                                ppg rawppgs = sensors.getnewppgseg(1800);
                                motion motions = sensors.getnewmotionseg(900);
//                                try {
//                                    Thread.currentThread().sleep(2000);
//                                } catch (InterruptedException e) {
//                                    e.printStackTrace();
//                                }
                                //上传人工确认后的短片段
                                segmentupload(RequestURL, motions, rawppgs, 10000, "s_segment", gesture_item);
//                                files.segmentupload_new(RequestURL,getApplicationContext(),motions, rawppgs, 10000, "s_segment", gesture_item);

                                Message msg = new Message();
                                msg.what = 1;
                                handler.sendMessage(msg);

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
                            if ((currenttime - starttime) < 7000) {
                                tips.setText("初始化中，请稍等！");
                                Log.e(">>>", "初始化，请稍等！");
                                uploadtag = 0;

                            } else {
                                if ((currenttime - ensuretime) < 7000) {
                                    tips.setText("录入中，请稍等！");
                                    Log.e(">>>", "录入中，请稍等！");
                                    uploadtag = 0;

                                } else {
                                    tips.setText("请录入手势！");
                                    Log.e(">>>", "请录入手势！");
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
                    }, 100, 1000);
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
        }
    };

    private void testtf() {
//        tf = new TensorFlowInferenceInterface(getAssets(),"ppg_model_weights.pb");
        try {
            //获取特征参数
            InputStream parameterinput = getAssets().open("stdpropara.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(parameterinput));
            String temp_sort1 = reader.readLine();
            String temp_sort2 = reader.readLine();
            String temp_scale_mean = reader.readLine();
            String temp_scale_scale = reader.readLine();
//            Log.e(">>>","tempscale_mean:"+temp_scale_mean);
            reader.close();
            parameterinput.close();

            String[] str_sort1 = temp_sort1.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] strsort2 = temp_sort2.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_mean = temp_scale_mean.replace("[", "").replace("]", "").replace(" ", "").split(",");
            String[] str_scale_scale = temp_scale_scale.replace("[", "").replace("]", "").replace(" ", "").split(",");

            Integer[] sort1 = nortools.strarraytointarray(str_sort1);
            Integer[] sort2 = nortools.strarraytointarray(strsort2);
            Double[] scale_mean = nortools.strarraytodoublearray(str_scale_mean);
            Double[] scale_scale = nortools.strarraytodoublearray(str_scale_scale);
//
//            Log.e(">>>","sort1:"+sort1[0]);
//            Log.e(">>>","sort2:"+sort2[0]);
//            Log.e(">>>","scale_mean:"+scale_mean[0]);
//            Log.e(">>>","scale_scale:"+scale_scale[0]);

            ArrayList<Integer> target = new ArrayList<Integer>();
            ArrayList<String> temp_feature = new ArrayList<String>();
            InputStream featurefile = getAssets().open("dataset/clc_1.csv");
            reader = new BufferedReader(new InputStreamReader(featurefile));
            String line;
            while ((line = reader.readLine()) != null) {
                temp_feature.add(line);
                target.add(0);
            }
            reader.close();
            featurefile.close();

            featurefile = getAssets().open("dataset/clx_1.csv");
            reader = new BufferedReader(new InputStreamReader(featurefile));
            while ((line = reader.readLine()) != null) {
                temp_feature.add(line);
                target.add(1);
            }
            reader.close();
            featurefile.close();


            String[][] str_feature = new String[temp_feature.size()][];
            for (int i = 0; i < temp_feature.size(); i++) {
                str_feature[i] = temp_feature.get(i).split(",");
            }
            Double[][] ppg_feature = new Double[temp_feature.size()][88];
            for (int i = 0; i < temp_feature.size(); i++) {
                Double[] temp = nortools.strarraytodoublearray(str_feature[i]);
                for (int j = 0; j < 88; j++) {
                    ppg_feature[i][j] = temp[j];
                }
            }

            Double[][] inform_feature = new Double[ppg_feature.length][30];
            for (int i = 0; i < temp_feature.size(); i++) {
                for (int j = 0; j < 30; j++) {
                    inform_feature[i][j] = ppg_feature[i][sort1[j]];
                }
            }

            Log.e(">>>", "feature:" + inform_feature[0][0]);
            ppg_feature = iatools.featurestd(inform_feature, scale_mean, scale_scale);
            Log.e(">>>", "feature:" + ppg_feature[0][0]);

            Integer[] targets = new Integer[target.size()];
            for (int i = 0; i < target.size(); i++) {
                targets[i] = target.get(i);
            }
            datapair pairs = iatools.create_pairs(ppg_feature, targets, 2);
            predict(pairs);
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

    private void loadModel(String model) {
        Interpreter tflite;
        Boolean load_result;
        try {
            tflite = new Interpreter(loadModelFile(model));
            Log.d(">>>", model + " model load success");
            tflite.setNumThreads(4);
            load_result = true;
        } catch (IOException e) {
            Log.d(">>>", model + " model load fail");
            load_result = false;
            e.printStackTrace();
        }
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(String model) throws IOException {
        Log.e(">>>", model + ".tflite");

        AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void predict(final datapair pairs) {
        //Runs inference in background thread
        new AsyncTask<Integer, Integer, Integer>() {

            @Override
            protected Integer doInBackground(Integer... params) {


                int datalen = pairs.x.size();
                double[][] temp_datax = new double[datalen][30];
                double[][] temp_datay = new double[datalen][30];
                Integer[] label = new Integer[datalen];
                for (int i = 0; i < datalen; i++) {
                    for (int j = 0; j < 30; j++) {
                        temp_datax[i][j] = pairs.x.get(i)[j];
                        temp_datay[i][j] = pairs.y.get(i)[j];
                        label[i] = pairs.label.get(i);
                    }
                }

                float[][]datax=new float[datalen][30];
                float[][]datay=new float[datalen][30];
                for (int i = 0; i < datalen; i++) {
                    for (int j = 0; j < 30; j++) {
                        datax[i][j]=(float)temp_datax[i][j];
                        datay[i][j]=(float)temp_datay[i][j];
                    }
                }

                Interpreter tflite;
                Boolean load_result;
                try {
                    tflite = new Interpreter(loadModelFile("based_model"));
                    Log.d(">>>", " model load success");
                    float[][] outPutsx = new float[datalen][256];//结果分类
                    float[][] outPutsy = new float[datalen][256];//结果分类

                    tflite.run(datax, outPutsx);
                    tflite.run(datay, outPutsy);
                    float []score=new float[datalen];
                    for(int i=0;i<datalen;i++){
                        float temp=0;
                        for(int j=0;j<256;j++){
                            temp=temp+(outPutsx[i][j]-outPutsy[i][j])*(outPutsx[i][j]-outPutsy[i][j]);
                        }
                        score[i]=(float)Math.sqrt(temp);
                    }
                    for (int i = 0; i < datalen; i++) {
                        Log.e(">>>", "score[i]:" + score[i] + "," + i);
                    }
                    double t=0.01;
                    while(t<3){
                        int tp=0;
                        int tn=0;
                        int fp=0;
                        int fn=0;
                        for(int j=0;j<datalen;j++){
                            if(score[j]<t){
                                if(1==label[j]){
                                    tp=tp+1;
                                }else{
                                    fp=fp+1;
                                }
                            }else{
                                if(1==label[j]){
                                    fn=fn+1;
                                }else{
                                    tn=tn+1;
                                }
                            }
                        }
//                        Log.e(">>>","tp:"+tp+",fp:"+fp+",frr:"+frr);

                        double accuracy=(double)(tp+tn)/(tp+tn+fp+fn);
                        double far=(double)(fp)/(fp+tn);
                        double frr=(double)(fn)/(fn+tp);
                        if ((frr<far) || (abs(frr-far)<0.02)){
                            Log.e(">>>","accuracy:"+accuracy+",far:"+far+",frr:"+frr);
                            break;
                        }
                        t=t+0.01;
                    }




                    load_result = true;
                } catch (IOException e) {
                    Log.d(">>>", " model load fail");
                    load_result = false;
                    e.printStackTrace();
                }


//                Interpreter.Options options = new Interpreter.Options();
//                options.setNumThreads(4);
//                tflite = new Interpreter(new File("stdpropara.txt"), options);
                //名称，输入张量，张量维度
//                    tf.feed("input",datax[0],1,30);
//                    //计算结果
//                    tf.run(new String[]{"output"});
//
//                    double [] outPuts1 = new double[256];//结果分类
//                    //将结果输出到output上
//                    tf.fetch("output",outPuts1);
//                    for(int i=0;i<256;i++){
//                        Log.e(">>>","outPuts[i]:"+outPuts1[i]+","+i);
//                    }
                return 0;
            }

        }.execute(0);

    }


    public void segmentupload(final String RequestURL,
                              final motion motions, final ppg ppgs, final int TIME_OUT, final String tag, final String item) {
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

                    data.put("username", "tzh_att");
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
//                    /**
//                     * 获取响应码  200=成功
//                     * 当响应成功，获取响应的流
//                     */
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
                } catch (MalformedURLException e) {
                    e.printStackTrace();
                    Message msg = new Message();
                    msg.what = 3;
                    handler.sendMessage(msg);
                } catch (IOException e) {
                    e.printStackTrace();
                    Message msg = new Message();
                    msg.what = 3;
                    handler.sendMessage(msg);
                } catch (JSONException e) {
                    e.printStackTrace();
                    Message msg = new Message();
                    msg.what = 3;
                    handler.sendMessage(msg);
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
