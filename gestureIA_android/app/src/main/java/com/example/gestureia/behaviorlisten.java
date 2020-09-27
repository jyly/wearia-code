package com.example.gestureia;


import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

public class behaviorlisten extends Service {
    private PowerManager.WakeLock wakeLock = null;

    private Sensorcontrol sensors = new Sensorcontrol();


    private Normal_tool nortools = new Normal_tool();
    private MAfind ma = new MAfind();
    private IAtool iatools = new IAtool();

    Timer timer = new Timer();

    @SuppressLint("InvalidWakeLockTag")
    @Override
    public void onCreate() {
        Log.i("Kathy", "onCreate - Thread ID = " + Thread.currentThread().getId());
        super.onCreate();

        iatools.energyopen(getApplicationContext());


//        launch();
        sensors.dataclear();
        sensors.StartSensorListening(getApplicationContext());


        timer.schedule(new TimerTask() {
            @Override
            public void run() {

                System.out.println("TimerTask");

                Log.e(">>>","ppg.size():"+sensors.getppgsize());

                if (sensors.getppgsize() > 2000 ) {

                    Ppg rawppgs = sensors.getnewppgseg(1800);
                    Motion motions = sensors.getnewmotionseg(900);

                    sensors.datadelete();

                    double[] tempx = nortools.arraytomatrix(rawppgs.x);
                    double[] tempy = nortools.arraytomatrix(rawppgs.y);

                    double[] orippgx = nortools.meanfilt(nortools.array_dataselect(tempx, tempx.length - 1000, 1000), 20);
                    int coarsetag = ma.coarse_grained_detect(orippgx);
                    Log.e(">>>","coarsetag:"+coarsetag);
                    if (1 == coarsetag) {

                        orippgx = nortools.meanfilt(tempx, 20);
                        double[] orippgy = nortools.meanfilt(tempy, 20);

                        double[] butterppgx = nortools.butterworth_highpass(orippgx, 200, 2);
                        double[] butterppgy = nortools.butterworth_highpass(orippgy, 200, 2);
                        Ppg ppgs = new Ppg();
                        ppgs.x = nortools.matrixtoarray(butterppgx);
                        ppgs.y = nortools.matrixtoarray(butterppgy);
                        ppgs = iatools.fastica(ppgs);
                        ppgs = iatools.machoice(ppgs);

                        int finetag = ma.fine_grained_segment(nortools.arraytomatrix(ppgs.x), 200, 1);

                        if (0 == finetag) {
                            System.out.println("当前片段不存在手势");
                        } else {
                            Featurecontrol singleFeaturecontrol = new Featurecontrol();
                            System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            ppgs = ma.setMAsegment(ppgs);
                            singleFeaturecontrol.ppg_feature(nortools.arraytomatrix(ppgs.x));
                            singleFeaturecontrol.ppg_feature(nortools.arraytomatrix(ppgs.y));
                        }
                    }

//                    stopService(new Intent(getBaseContext(), sensorlisten.class));
                }
            }
        }, 100, 1000);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
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
//        super.onDestroy();
    }



}
