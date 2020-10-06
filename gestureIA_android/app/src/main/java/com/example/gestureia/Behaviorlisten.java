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
    private Energycontrol energycontrol = null;
    private Sensorcontrol sensors = null;
    private Baedmodel basedmodel = null;
    Timer timer = null;
    private int sleepcount = 0;


    @SuppressLint("InvalidWakeLockTag")
    @Override
    public void onCreate() {
        Log.i("Kathy", "onCreate - Thread ID = " + Thread.currentThread().getId());
        super.onCreate();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        sensors = new Sensorcontrol();
        energycontrol = new Energycontrol();
        basedmodel = new Baedmodel();

        energycontrol.energyclose();
        energycontrol.energyopen(getApplicationContext());
//        launch();
        sensors.dataclear();
        sensors.StopSensorListening();
        sensors.StartSensorListening(getApplicationContext());
        basedmodel.readmodelpara(getApplicationContext());
        basedmodel.readbasedfeature(getApplicationContext());
        timer = new Timer();

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (sleepcount > 0) {
                    Log.e(">>>", "sleepcount:" + sleepcount);
                    sleepcount--;
//                    if (sensors.getppgsize() > 4000&&0==readlock) {
////                        sensors.datadelete();
//                    }
                }

                Log.e(">>>", "ppg.size():" + sensors.getppgsize());

                if (sensors.getppgsize() > 2000 && sleepcount == 0) {
                    Ppg ppgs = sensors.getnewppgseg(1800);
                    Motion motion = sensors.getnewmotionseg(900);
                    System.gc();

//                    sensors.datadelete();

                    Normal_tool nortools = new Normal_tool();
                    MAfind ma = new MAfind();
                    ppgs.x = nortools.meanfilt(ppgs.x, 20);
                    ppgs.y = nortools.meanfilt(ppgs.y, 20);

                    int coarsetag = ma.coarse_grained_detect(ppgs.x);
                    Log.e(">>>", "coarsetag:" + coarsetag);
                    if (1 == coarsetag) {
                        //对原始的ppg型号做butterworth提取
                        Ppg butterppg = new Ppg();
                        butterppg.x = nortools.butterworth_highpass(ppgs.x, 200, 2);
                        butterppg.y = nortools.butterworth_highpass(ppgs.y, 200, 2);
                        // 做快速主成分分析
                        IAtool iatools = new IAtool();
                        Ppg icappg = iatools.fastica(butterppg);
                        // 根据峰值判断那条手势信号和脉冲信号
                        icappg = iatools.machoice(icappg);
                        iatools = null;
                        //细粒度手势分析，判断手势区间
//                        int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
                        int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1);

                        if (0 == finetag) {
                            Log.e(">>>", "当前片段不存在手势");
//                            System.out.println("当前片段不存在手势");
                        } else {
                            Log.e(">>>", "手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            //特征提取
                            ppgs.x=nortools.innerscale(ppgs.x);
                            ppgs.y=nortools.innerscale(ppgs.y);

                            ppgs = ma.setppgsegment(ppgs);
                            butterppg = ma.setppgsegment(butterppg);
                            icappg = ma.setppgsegment(icappg);
                            motion = ma.setmotionsegment(motion);

                            Featurecontrol featurecontrol = new Featurecontrol();
                            double[] featureset = featurecontrol.return_feature(ppgs, motion, butterppg, icappg);
                            featurecontrol = null;
//                            for(int i=0;i<temp.length;i++){
//                                System.out.print(temp[i]+",");
//                            }
                            //特征过滤及预处理
                            float[][] final_feature= basedmodel.dataprocess(featureset);
                            int predittag = basedmodel.behavior_predit(basedmodel.final_feature, final_feature);
                            featureset=null;
                            final_feature=null;

//                            if (predittag == 1) {
//                                sleepcount = 8;
//                            }
                        }
                        butterppg=null;
                        icappg=null;
                        motion=null;
                    }
//                    stopService(new Intent(getBaseContext(), sensorlisten.class));
                    ppgs=null;
                    motion=null;
                    nortools = null;
                    ma = null;
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
        energycontrol.energyclose();
        timer.cancel();
        sensors = null;
        energycontrol = null;
        timer = null;
        basedmodel = null;
        super.onDestroy();
    }
}
