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
    private Baedmodel basedmodel=new Baedmodel();



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
        basedmodel.readmodelpara(getApplicationContext());
        basedmodel.readbasedfeature(getApplicationContext());

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
                            Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            //特征提取
                            ArrayList<Double> samplefeature= new ArrayList<Double>();
                            ppgs.x = nortools.matrixtoarray(nortools.array_dataselect(orippgx, 300, orippgx.length - 300));
                            ppgs.y = nortools.matrixtoarray(nortools.array_dataselect(orippgy, 300, orippgy.length - 300));
                            ppgs = ma.setppgsegment(ppgs);
                            int datalen=motion.accx.size();
                            motion.accx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accx),150,  datalen- 150));
                            motion.accy = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accy),150, datalen - 150));
                            motion.accz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accz),150, datalen - 150));
                            motion.gyrx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrx),150, datalen - 150));
                            motion.gyry = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyry),150, datalen - 150));
                            motion.gyrz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrz),150, datalen - 150));

                            motion=ma.setmotionsegment(motion);
                            samplefeature=featurecontrol.return_feature(ppgs,motion);
                            //特征过滤及预处理
                            float[] inform_feature = new float[60];
                            for (int i = 0; i < 30; i++) {
                                inform_feature[i] = (float) (double) samplefeature.get(basedmodel.sort1[i]);
                                inform_feature[i + 30] = (float) (double) samplefeature.get(basedmodel.sort2[i] + 84);
                            }
                            inform_feature = iatools.featurestd(inform_feature, basedmodel.scale_mean, basedmodel.scale_scale);
                            float[] ppg_feature = new float[30];
                            float[] motion_feature = new float[30];
                            for (int i = 0; i < 30; i++) {
                                ppg_feature[i] = inform_feature[i];
                                motion_feature[i] = inform_feature[i + 30];
                            }
                            float[][] temp_final_feature = new float[2][];
                            temp_final_feature[0] = siamese.sample_feature(basedmodel.ppg_tflite, ppg_feature);
                            temp_final_feature[1] = siamese.sample_feature(basedmodel.motion_tflite, motion_feature);

                            int predittag=siamese.behavior_predit(basedmodel.final_feature,temp_final_feature);
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
}
