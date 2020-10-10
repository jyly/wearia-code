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

        energycontrol.energyopen(getApplicationContext());
        sensors.StartSensorListening(getApplicationContext());
        basedmodel.readmodelpara(getApplicationContext());
        timer = new Timer();

        basedmodel.readmulfeature(getApplicationContext());
//        basedmodel.readfeature(getApplicationContext());

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (sleepcount > 0) {
                    Log.e(">>>", "sleepcount:" + sleepcount);
                    sleepcount--;
                }

                Log.e(">>>", "ppg.newindex:" + sensors.getppgsize());

                if (1==sensors.maxflag && sleepcount == 0) {
                    Ppg ppgs = sensors.getnewppgseg(1600);
                    Motion motions = sensors.getnewmotionseg(800);

                    Normal_tool nortools = new Normal_tool();
                    MAfind ma = new MAfind();
                    Ppg orippg=new Ppg();
                    orippg.x = nortools.meanfilt(ppgs.x, 20);
                    orippg.y = nortools.meanfilt(ppgs.y, 20);

//                    int coarsetag = ma.coarse_grained_detect(orippg.x);
                    int coarsetag = 1;
                    Log.e(">>>", "coarsetag:" + coarsetag);
                    if (1 == coarsetag) {
                        //对原始的ppg型号做butterworth提取
                        Ppg butterppg = new Ppg();
                        butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
                        butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);
                        // 做快速主成分分析
                        IAtool iatools = new IAtool();
                        Ppg icappg = iatools.fastica(butterppg);
                        // 根据峰值判断那条手势信号和脉冲信号
                        icappg = iatools.machoice(icappg);
                        iatools = null;
                        //细粒度手势分析，判断手势区间
//                        int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
                        int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5);

                        if (0 == finetag) {
                            Log.e(">>>", "当前片段不存在手势");
//                            System.out.println("当前片段不存在手势");
                        } else {
                            Log.e(">>>", "手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            //特征提取
                            orippg.x = nortools.innerscale(orippg.x);
                            orippg.y = nortools.innerscale(orippg.y);

                            orippg = ma.setppgsegment(orippg);
                            butterppg = ma.setppgsegment(butterppg);
                            Motion motion = ma.setmotionsegment(motions);

                            scorerun(butterppg,motion);
                            motion=null;
                            orippg=null;
                        }
                        butterppg=null;
                        icappg=null;

                    }
                    orippg=null;
                    ppgs=null;
                    motions=null;
                    nortools = null;
                    ma = null;
                }
                System.gc();

            }
        }, 100, 1000);
        return super.onStartCommand(intent, flags, startId);
    }


    public void scorerun(final Ppg ppg,final Motion motion ){
        new Thread(new Runnable() {
            @Override
            public void run() {
//                            double[][] madata = new double[8][300];
//                            madata[0] = orippg.x;
//                            madata[1] = orippg.y;
//                            madata[2] = motion.accx;
//                            madata[3] = motion.accy;
//                            madata[4] = motion.accz;
//                            madata[5] = motion.gyrx;
//                            madata[6] = motion.gyry;
//                            madata[7] = motion.gyrz;
//                            float[] final_feature= basedmodel.dataprocess(madata);

                Featurecontrol featurecontrol=new Featurecontrol();
                double[] featureset = featurecontrol.return_feature(ppg, motion);
                featurecontrol = null;
                if(featureset.length!=0){
                    float[][] final_feature= basedmodel.featureprocess(featureset);
                    int predittag = basedmodel.behavior_predit(final_feature);
                    final_feature=null;
                }

//                            if (predittag == 1) {
//                                sleepcount = 8;
//                            }

                featureset=null;
            }}).start();
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
        sleepcount=0;
        super.onDestroy();
    }
}
