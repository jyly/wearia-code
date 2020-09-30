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
    private Energycontrol energycontrol = new Energycontrol();
    private Sensorcontrol sensors = new Sensorcontrol();
    private Normal_tool nortools = new Normal_tool();
    private MAfind ma = new MAfind();
    private IAtool iatools = new IAtool();
    private Siamese_model siamese = new Siamese_model();
    private Baedmodel basedmodel = new Baedmodel();


    Timer timer = new Timer();
    private int sleepcount = 0;

    @SuppressLint("InvalidWakeLockTag")
    @Override
    public void onCreate() {
        Log.i("Kathy", "onCreate - Thread ID = " + Thread.currentThread().getId());
        super.onCreate();

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        energycontrol.energyclose();
        energycontrol.energyopen(getApplicationContext());
//        launch();
        sensors.dataclear();
        sensors.StopSensorListening();
        sensors.StartSensorListening(getApplicationContext());
        basedmodel.readmodelpara(getApplicationContext());
        basedmodel.readbasedfeature(getApplicationContext());

        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (sleepcount > 0) {
                    Log.e(">>>", "sleepcount:" + sleepcount);
                    sleepcount--;
                    if (sensors.getppgsize() > 2000) {
                        sensors.datadelete();
                    }
                }
//                System.out.println("TimerTask");

                Log.e(">>>", "ppg.size():" + sensors.getppgsize());

                if (sensors.getppgsize() > 2000 && sleepcount == 0) {

//                    Ppg rawppgs = sensors.getnewppgseg(1800);
//                    Motion motion = sensors.getnewmotionseg(900);

                    Ppg ppgs = sensors.getnewppgseg(1800);
                    Motion motion = sensors.getnewmotionseg(900);

                    ppgs.x = nortools.meanfilt(ppgs.x, 20);
                    ppgs.y = nortools.meanfilt(ppgs.y, 20);
                    sensors.datadelete();

                    int coarsetag = ma.coarse_grained_detect(ppgs.x);
                    Log.e(">>>", "coarsetag:" + coarsetag);
                    if (1 == coarsetag) {

                        Ppg butterppg = new Ppg();
                    	//对原始的ppg型号做butterworth提取
                        butterppg.x = nortools.butterworth_highpass(ppgs.x, 200, 2);
                        butterppg.y = nortools.butterworth_highpass(ppgs.y, 200, 2);

                        int inter=600;
                        butterppg.x = nortools.array_dataselect(butterppg.x, inter, butterppg.x.length - inter);
                        butterppg.y = nortools.array_dataselect(butterppg.y, inter, butterppg.y.length - inter);
                        // 做快速主成分分析
                        Ppg icappg = iatools.fastica(butterppg);
                        // 根据峰值判断那条手势信号和脉冲信号
                        icappg = iatools.machoice(icappg);
                        //细粒度手势分析，判断手势区间
                        int finetag = ma.fine_grained_segment(icappg.x, 200, 1);

                        if (0 == finetag) {
                            Log.e(">>>", "当前片段不存在手势");
//                            System.out.println("当前片段不存在手势");
                        } else {
                            Log.e(">>>", "手势点：" + ma.pointstartindex + " " + ma.pointendindex);
                            //特征提取
                            ppgs.x = nortools.array_dataselect(ppgs.x, inter, ppgs.x.length - inter);
                            ppgs.y = nortools.array_dataselect(ppgs.y, inter, ppgs.y.length - inter);

                            ppgs = ma.setppgsegment(ppgs);
                            butterppg = ma.setppgsegment(butterppg);
                            icappg = ma.setppgsegment(icappg);
                            int datalen=motion.accx.length;
                            motion.accx = nortools.array_dataselect(motion.accx,inter/2,  datalen- inter/2);
                            motion.accy = nortools.array_dataselect(motion.accy,inter/2, datalen - inter/2);
                            motion.accz = nortools.array_dataselect(motion.accz,inter/2, datalen - inter/2);
                            motion.gyrx = nortools.array_dataselect(motion.gyrx,inter/2, datalen - inter/2);
                            motion.gyry = nortools.array_dataselect(motion.gyry,inter/2, datalen - inter/2);
                            motion.gyrz = nortools.array_dataselect(motion.gyrz,inter/2, datalen - inter/2);

                            motion=ma.setmotionsegment(motion);
                            ma=null;
                            Featurecontrol featurecontrol = new Featurecontrol();

                            double[] temp=featurecontrol.return_feature(ppgs,motion,butterppg,icappg);

                            //特征过滤及预处理
                            int featurelen=60;
                            float[] inform_feature = new float[featurelen*2];
                            for (int i = 0; i < featurelen; i++) {
                                inform_feature[i] = (float) (double) temp[basedmodel.sort1[i]];
                                inform_feature[i + featurelen] = (float) (double) temp[basedmodel.sort2[i] + 252];
                            }
                            inform_feature = featurecontrol.featurestd(inform_feature, basedmodel.scale_mean, basedmodel.scale_scale);
                            featurecontrol=null;
                            float[] ppg_feature = new float[featurelen];
                            float[] motion_feature = new float[featurelen];
                            for (int i = 0; i < featurelen; i++) {
                                ppg_feature[i] = inform_feature[i];
                                motion_feature[i] = inform_feature[i + featurelen];
                            }

                            float[][] final_feature = new float[2][];
                            Siamese_model siamese = new Siamese_model();
                            final_feature[0] = siamese.sample_feature(basedmodel.ppg_tflite, ppg_feature);
                            final_feature[1] = siamese.sample_feature(basedmodel.motion_tflite, motion_feature);

                            int predittag = siamese.behavior_predit(basedmodel.final_feature, final_feature);
                            siamese=null;

                            if (predittag == 1) {
                                sleepcount = 8;
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
        energycontrol.energyclose();
        timer.cancel();
        super.onDestroy();
    }
}
