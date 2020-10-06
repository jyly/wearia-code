package com.example.gestureia;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;
import java.util.List;

public class Sensorcontrol {

    private SensorManager mSensorManager;

    float[] ppgx = new float[6000];
    float[] ppgy = new float[6000];
    long[] psensortime = new long[6000];
    int pi = 0;

    float[] accx = new float[4000];
    float[] accy = new float[4000];
    float[] accz = new float[4000];
    long[] asensortime = new long[4000];
    int ai = 0;

    float[] gyrx = new float[4000];
    float[] gyry = new float[4000];
    float[] gyrz = new float[4000];
    long[] gsensortime = new long[4000];
    int gi = 0;

    public void StartSensorListening(Context context) {

        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(65572), 0);//Heart Rate PPG Raw Data 200hz
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(1), 0);//Accelerometer 100hz
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(4), 0);//Gyroscope 100hz
//        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(5), 0);//light
        List<Sensor> sensorList = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        Log.e(">>>", "列出当前设备所有的传感器信息");
        for (Sensor sensor : sensorList) {
            Log.e("List sensors", "Name: " + sensor.getName() + " /Type_String: " + sensor.getStringType() + " /Type_number: " + sensor.getType());
        }
        sensorList=null;
    }

    public void StopSensorListening() {
        if (mSensorManager != null) {
            mSensorManager.unregisterListener(listener);
            mSensorManager = null;
        }
    }

    private SensorEventListener listener = new SensorEventListener() {
        @Override
        public void onAccuracyChanged(Sensor sensor, int i) {
        }
        public void onSensorChanged(SensorEvent e) {
            switch (e.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:   //加速度传感器
//                    Log.d("Test", "Got the ACCELEROMETER : " + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]) + "," + String.valueOf(e.values[2]));
                    accx[ai] = e.values[0];
                    accy[ai] = e.values[1];
                    accz[ai] = e.values[2];
                    asensortime[ai] = System.currentTimeMillis();
                    ai = ai + 1;
                    break;
                case Sensor.TYPE_GYROSCOPE:     //陀螺传感器
//                    Log.d("Test", "Got the GYROSCOPE : " + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]) + "," + String.valueOf(e.values[2]));
                    gyrx[gi] = e.values[0];
                    gyry[gi] = e.values[1];
                    gyrz[gi] = e.values[2];
                    gsensortime[gi] = System.currentTimeMillis();
                    gi = gi + 1;
                    break;
                case 65572:   //ppg传感器
//                    Log.d("Test", "raw ppg : "+e.values.length+"," + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]));
//                    String []indicnum=String.valueOf(e.values[0]).split("E");
//                    Log.d("Test", "indicnum : "+ indicnum[0] + "," + (50+float.valueOf(indicnum[1])));
//
//                    float tempx=float.valueOf(indicnum[0])*Math.pow(10, (50+float.valueOf(indicnum[1])))/2;
//                    indicnum=String.valueOf(e.values[1]).split("E");
//                    float tempy=float.valueOf(indicnum[0])*Math.pow(10, (50+float.valueOf(indicnum[1])))/2;
//                    Log.d("Test", "raw ppg : "+e.values.length+"," + tempx + "," + tempy);
//                    for(int i=0;i<16;i++){
//                        Log.e(">>>>","e.value:"+i+","+e.values[i]);
//                    }
                    ppgx[pi] = e.values[0];
                    ppgy[pi] = e.values[1];
                    psensortime[pi] = System.currentTimeMillis();
                    pi = pi + 1;
                    break;
//                case 5:   //光学ppg传感器
//                    //大于100基本就是没在使用
//                    Log.d("Test", "light : " +e.values.length+","+ String.valueOf(e.values[0]));
//                    break;
            }
            if(pi>4000||ai>2000||gi>2000){
                datadelete();
            }
        }
    };

    public void dataclear() {
        ppgx = null;
        ppgy = null;
        psensortime = null;
        accx = null;
        accy = null;
        accz = null;
        asensortime = null;
        gyrx = null;
        gyry = null;
        gyrz = null;
        gsensortime = null;

        ppgx = new float[6000];
        ppgy = new float[6000];
        psensortime = new long[6000];
        pi = 0;
        accx = new float[4000];
        accy = new float[4000];
        accz = new float[4000];
        asensortime = new long[4000];
        ai = 0;
        gyrx = new float[4000];
        gyry = new float[4000];
        gyrz = new float[4000];
        gsensortime = new long[4000];
        gi = 0;

    }

    public Ppg getnewppgseg(int lens) {
        int length = pi;
        if (lens > length - 100) {
            lens = length - 100;
        }
        double[] x = new double[lens];
        double[] y = new double[lens];
        long[] timestamp = new long[lens];
        for (int i = 0; i < lens; i++) {
            if (ppgx[i + length - lens - 10] != 0 && ppgy[i + length - lens - 10] != 0) {
                x[i] = ppgx[i + length - lens - 10];
                y[i] = ppgy[i + length - lens - 10];
                timestamp[i] = psensortime[i + length - lens - 10];
            }
        }
        Ppg ppgs = new Ppg(x, y, timestamp);
        return ppgs;
    }

    public Ppg getnewppgseg() {
        return getnewppgseg(pi);
    }

    public Motion getnewmotionseg(int lens) {
        int length = ai;
        if (lens > length - 50) {
            lens = length - 50;
        }
        double[] acc_x = new double[lens];
        double[] acc_y = new double[lens];
        double[] acc_z = new double[lens];
        long[] acctime = new long[lens];

        for (int i = 0; i < lens; i++) {
            if (accx[i + length - lens - 10] != 0 && accy[i + length - lens - 10] != 0 && accz[i + length - lens - 10] != 0) {
                acc_x[i] = accx[i + length - lens - 10];
                acc_y[i] = accy[i + length - lens - 10];
                acc_z[i] = accz[i + length - lens - 10];
                acctime[i] = asensortime[i + length - lens - 10];
            }
        }

        length = gi;
        if (lens > length - 50) {
            lens = length - 50;
        }
        double[] gyr_x = new double[lens];
        double[] gyr_y = new double[lens];
        double[] gyr_z = new double[lens];
        long[] gyrtime = new long[lens];

        for (int i = 0; i < lens; i++) {
            if (gyrx[i + length - lens - 10] != 0 && gyry[i + length - lens - 10] != 0 && gyrz[i + length - lens - 10] != 0) {
                gyr_x[i] = gyrx[i + length - lens - 10];
                gyr_y[i] = gyry[i + length - lens - 10];
                gyr_z[i] = gyrz[i + length - lens - 10];
                gyrtime[i] = gsensortime[i + length - lens - 10];
            }
        }
        Motion motions = new Motion(acc_x, acc_y, acc_z, acctime, gyr_x, gyr_y, gyr_z, gyrtime);
        return motions;
    }

    public void datadelete() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                //        Log.e(">>>","data_delete");
                if (pi > 4000) {
                    int lens = pi;
                    for (int i = 0; i < 2000; i++) {
                        ppgx[i] = ppgx[lens - 2000 + i];
                        ppgy[i] = ppgy[lens - 2000 + i];
                        psensortime[i] = psensortime[lens - 2000 + i];
                    }
                    pi = 2000;
                }

                if (ai > 2000) {
                    int lens = ai;
                    for (int i = 0; i < 1000; i++) {
                        accx[i] = accx[lens - 1000 + i];
                        accy[i] = accy[lens - 1000 + i];
                        accz[i] = accz[lens - 1000 + i];
                        asensortime[i] = asensortime[lens - 1000 + i];
                    }
                    ai = 1000;
                }

                if (gi > 2000) {
                    int lens = gi;
                    for (int i = 0; i < 1000; i++) {
                        gyrx[i] = gyrx[lens - 1000 + i];
                        gyry[i] = gyry[lens - 1000 + i];
                        gyry[i] = gyry[lens - 1000 + i];
                        gsensortime[i] = gsensortime[lens - 1000 + i];
                    }
                    gi = 1000;
                }
            }
        }).start();
    }


    public int getppgsize() {
        return pi;
    }

    public int getaccsize() {
        return ai;
    }

    public int getgyrsize() {
        return gi;
    }
}
