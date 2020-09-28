package com.example.gestureia;


import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.os.IBinder;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import androidx.annotation.Nullable;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

public class Behaviorlisten extends Service {

    private Sensorcontrol sensors = new Sensorcontrol();
    private Featurecontrol featurecontrol=new Featurecontrol();
    private Normal_tool nortools = new Normal_tool();
    private MAfind ma = new MAfind();
    private IAtool iatools = new IAtool();
    private Siamese_model siamese=new Siamese_model();

    private Integer[] sort1 = null;
    private Integer[] sort2 = null;
    private Double[] scale_mean = null;
    private Double[] scale_scale = null;
    private Interpreter ppg_tflite = null;
    private Interpreter motion_tflite = null;

    private Double[][] final_feature = null;

    Timer timer = new Timer();
    private int sleepcount=0;
    @SuppressLint("InvalidWakeLockTag")
    @Override
    public void onCreate() {
        Log.i("Kathy", "onCreate - Thread ID = " + Thread.currentThread().getId());
        super.onCreate();

    }


    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        iatools.energyclose();
        iatools.energyopen(getApplicationContext());
//        launch();
        sensors.dataclear();
        sensors.StopSensorListening();
        sensors.StartSensorListening(getApplicationContext());
        readmodelpara();
        readbasedfeature();


        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if(sleepcount>0){
                    Log.e(">>>", "sleepcount:" +sleepcount);
                    sleepcount--;
                    if(sensors.getppgsize() > 2000){
                        sensors.datadelete();
                    }
                }
//                System.out.println("TimerTask");

                Log.e(">>>", "ppg.size():" + sensors.getppgsize());

                if (sensors.getppgsize() > 2000&&sleepcount==0) {

                    Ppg rawppgs = sensors.getnewppgseg(1800);
                    Motion motion = sensors.getnewmotionseg(900);


                    double[] orippgx = nortools.meanfilt(nortools.arraytomatrix(rawppgs.x), 20);
                    double[] orippgy = nortools.meanfilt(nortools.arraytomatrix(rawppgs.y), 20);
                    sensors.datadelete();

                    int coarsetag = ma.coarse_grained_detect(orippgx);
                    Log.e(">>>", "coarsetag:" + coarsetag);
                    if (1 == coarsetag) {

                        double[] butterppgx = nortools.butterworth_highpass(orippgx, 200, 2);
                        double[] butterppgy = nortools.butterworth_highpass(orippgy, 200, 2);

                        Ppg butterppg = new Ppg();
                        butterppg.x = nortools.matrixtoarray(nortools.array_dataselect(butterppgx, 300, butterppgx.length - 300));
                        butterppg.y = nortools.matrixtoarray(nortools.array_dataselect(butterppgy, 300, butterppgx.length - 300));
                        // 做快速主成分分析
                        butterppg = iatools.fastica(butterppg);
                        // 根据峰值判断那条手势信号和脉冲信号
                        butterppg = iatools.machoice(butterppg);
                        //细粒度手势分析，判断手势区间
                        int finetag = ma.fine_grained_segment(nortools.arraytomatrix(butterppg.x), 200, 1);
                        Ppg ppgs=new Ppg();
                        if (0 == finetag) {
                            System.out.println("当前片段不存在手势");
                        } else {
                            //特征提取
                            ArrayList<Double> sample= new ArrayList<Double>();
                            ArrayList<Double> motionfeature= new ArrayList<Double>();
                            ppgs.x = nortools.matrixtoarray(nortools.array_dataselect(orippgx, 300, orippgx.length - 300));
                            ppgs.y = nortools.matrixtoarray(nortools.array_dataselect(orippgy, 300, orippgy.length - 300));
                            Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            ppgs = ma.setMAsegment(ppgs);
                            motion.accx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accx),150, motion.accx.size() - 150));
                            motion.accy = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accy),150, motion.accx.size() - 150));
                            motion.accz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accz),150, motion.accx.size() - 150));
                            motion.gyrx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrx),150, motion.gyrx.size() - 150));
                            motion.gyry = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyry),150, motion.gyrx.size() - 150));
                            motion.gyrz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrz),150, motion.gyrx.size() - 150));
                            motion=ma.setmotionsegment(motion);







                            //特征过滤及预处理
                            float[] inform_feature = new float[60];
                            for (int i = 0; i < 30; i++) {
                                inform_feature[i] = (float)(double)ppgfeature.get(sort1[i]);
                            }
                            for (int i = 0; i < 30; i++) {
                                inform_feature[i+30] = (float)(double)motionfeature.get(sort2[i]);
                            }
                            inform_feature = iatools.featurestd(inform_feature, scale_mean, scale_scale);
                            float []ppg_feature=new float[30];
                            for (int i = 0; i < 30; i++) {
                                ppg_feature[i] = inform_feature[i];
                            }
                            float []motion_feature=new float[30];
                            for (int i = 0; i < 30; i++) {
                                motion_feature[i] = inform_feature[i+30];
                            }
                            //                            float[] temp_final_feature =siamese.sample_feature(tflite, inform_feature);
                            float[] temp_ppg_feature =siamese.ppg_feature(ppg_tflite, ppg_feature);
                            float[] temp_motion_feature =siamese.motion_feature(motion_tflite, motion_feature);
                            int predittag=siamese.behavior_predit(final_feature,temp_final_feature);
                            if (predittag==1){
                                sleepcount=8;
                            }


                        }
                    }
//                    stopService(new Intent(getBaseContext(), sensorlisten.class));
                }
            }
        }, 100, 1000);
        return super.onStartCommand(intent, flags, startId);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        Log.i("Kathy", "onBind - Thread ID = " + Thread.currentThread().getId());
        return null;
    }

    @Override
    public void onDestroy() {
        Log.i("Kathy", "onDestroy - Thread ID = " + Thread.currentThread().getId());
        sensors.StopSensorListening();
        iatools.energyclose();
        timer.cancel();
        super.onDestroy();
    }
    private void readmodelpara() {
        try {
            ppg_tflite = new Interpreter(loadModelFile("ppg_based_model"));
            motion_tflite = new Interpreter(loadModelFile("motion_based_model"));
            InputStream parameterinput = getAssets().open("stdpropara.txt");
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
            Integer[] sort2 = nortools.strarraytointarray(strsort2);
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
                //提取128位最终的向量
                int listnum = 0;
                if (temp_feature.size() > 5) {
                    listnum = 5;
                } else {
                    listnum = temp_feature.size();
                }
                String[][] str_feature = new String[listnum][];
                final_feature = new Double[listnum][128];
                for (int i = 0; i < listnum; i++) {
                    str_feature[i] = temp_feature.get(i).split(",");
                }
                for (int i = 0; i < listnum; i++) {
                    for (int j = 0; j < 128; j++) {
                        final_feature[i][j] = Double.parseDouble(str_feature[i][j]);
                    }
                }

            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
