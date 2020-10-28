package com.example.gestureia;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import java.util.List;

public class Sensorcontrol {

    private SensorManager mSensorManager = null;

    float[] ppgx = null;
    float[] ppgy = null;
    long[] psensortime = null;
    int pi = 0;

    float[] accx = null;
    float[] accy = null;
    float[] accz = null;
    long[] asensortime = null;
    int ai = 0;

    float[] gyrx = null;
    float[] gyry = null;
    float[] gyrz = null;
    long[] gsensortime = null;
    int gi = 0;

    int maxflag = 0;

    public void StartSensorListening(Context context) {
        ppgx = new float[6000];
        ppgy = new float[6000];
        psensortime = new long[6000];
        pi = 0;

        accx = new float[3000];
        accy = new float[3000];
        accz = new float[3000];
        asensortime = new long[3000];
        ai = 0;

        gyrx = new float[3000];
        gyry = new float[3000];
        gyrz = new float[3000];
        gsensortime = new long[3000];
        gi = 0;

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
        sensorList = null;
    }

    public void StopSensorListening() {
        pi = 0;
        ai = 0;
        gi = 0;
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
                    if (e.values != null) {
                        accx[ai] = e.values[0];
                        accy[ai] = e.values[1];
                        accz[ai] = e.values[2];
                        asensortime[ai] = System.currentTimeMillis();
                        if (ai == 2999) {
                            ai = 0;
                        } else {
                            ai = ai + 1;
                        }
                    }


                    break;
                case Sensor.TYPE_GYROSCOPE:     //陀螺传感器
//                    Log.d("Test", "Got the GYROSCOPE : " + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]) + "," + String.valueOf(e.values[2]));
                    if (e.values != null) {
                        gyrx[gi] = e.values[0];
                        gyry[gi] = e.values[1];
                        gyrz[gi] = e.values[2];
                        gsensortime[gi] = System.currentTimeMillis();
                        if (gi == 2999) {
                            gi = 0;
                        } else {
                            gi = gi + 1;
                        }
                    }

                    break;
                case 65572:   //ppg传感器
//                    Log.d("Test", "raw ppg : " + e.values.length + "," + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]));
                    if (e.values != null) {
                        if (e.values[0] != 0) {
                            //                        Log.d("Test", "indicnum : " + indicnum[0] + "," + (46 + Float.valueOf(indicnum[1])));
//                        String[] indicnum = String.valueOf(e.values[0]).split("E");
//                        float tempx = Float.valueOf(indicnum[0]) * (float)Math.pow(10, (46 + Float.valueOf(indicnum[1])));
//                        indicnum = String.valueOf(e.values[1]).split("E");
//                        float tempy = Float.valueOf(indicnum[0]) * (float)Math.pow(10, (46 + Float.valueOf(indicnum[1])));
//                        Log.d("Test", "tempx&tempy : " + e.values.length + "," + tempx + "," + tempy);
//                        String[] indicnum = String.valueOf(e.values[0]).split("E");
//                        ppgx[pi] = Float.valueOf(indicnum[0]) * (float) Math.pow(10, (46 + Float.valueOf(indicnum[1])));
//                        indicnum = String.valueOf(e.values[1]).split("E");
//                        ppgy[pi] = Float.valueOf(indicnum[0]) * (float) Math.pow(10, (46 + Float.valueOf(indicnum[1])));

                            ppgx[pi] = e.values[0];
                            ppgy[pi] = e.values[1];
                            psensortime[pi] = System.currentTimeMillis();

                            if (pi == 5999) {
                                pi = 0;
                            } else {
                                pi = pi + 1;
                            }
                        }
                    }
                    break;
//                case 5:   //光学ppg传感器
//                    //大于100基本就是没在使用
//                    Log.d("Test", "light : " +e.values.length+","+ String.valueOf(e.values[0]));
//                    break;
            }
            if (0 == maxflag) {
                if (2000 == pi) {
                    maxflag = 1;
                }
            }
        }
    };


    public Ppg getnewppgseg(int lens) {
        int length = pi;
        Ppg ppgs = new Ppg(lens);
        for (int i = 0; i < lens; i++) {
//            int iters = (6000 + i + length - lens - 10) % 6000;
            int iters = i + length - lens - 10;
            if (iters < 0) {
                iters = iters + 6000;
            }
            ppgs.x[i] = ppgx[iters];
            ppgs.y[i] = ppgy[iters];
            ppgs.timestamps[i] = psensortime[iters];
        }
        return ppgs;
    }

    public Ppg getnewppgseg() {
        return getnewppgseg(pi);
    }

    public Motion getnewmotionseg(int lens) {

        int iters;
        Motion motions = new Motion(lens);

        int length = ai;
        for (int i = 0; i < lens; i++) {
            iters = i + length - lens - 10;
            if (iters < 0) {
                iters = iters + 3000;
            }
            motions.accx[i] = accx[iters];
            motions.accy[i] = accy[iters];
            motions.accz[i] = accz[iters];
            motions.acctimestamps[i] = asensortime[iters];
        }
        length = gi;
        for (int i = 0; i < lens; i++) {
            iters = i + length - lens - 10;
            if (iters < 0) {
                iters = iters + 3000;
            }
            motions.gyrx[i] = gyrx[iters];
            motions.gyry[i] = gyry[iters];
            motions.gyrz[i] = gyrz[iters];
            motions.gyrtimestamps[i] = gsensortime[iters];
        }
        return motions;
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
