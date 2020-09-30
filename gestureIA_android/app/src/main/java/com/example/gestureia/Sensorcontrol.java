package com.example.gestureia;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class Sensorcontrol {
    private SensorManager mSensorManager;
    private ArrayList<Double> ppgx = new ArrayList<Double>();
    private ArrayList<Double> ppgy = new ArrayList<Double>();

    private ArrayList<Double> accx = new ArrayList<Double>();
    private ArrayList<Double> accy = new ArrayList<Double>();
    private ArrayList<Double> accz = new ArrayList<Double>();

    private ArrayList<Double> gyrx = new ArrayList<Double>();
    private ArrayList<Double> gyry = new ArrayList<Double>();
    private ArrayList<Double> gyrz = new ArrayList<Double>();

    private ArrayList<Long> ppgtimestamps = new ArrayList<>();
    private ArrayList<Long> acctimestamps = new ArrayList<>();
    private ArrayList<Long> gyrtimestamps = new ArrayList<>();
    public List<Sensor> sensorList;

    public void StartSensorListening(Context context) {

        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(65572), 0);//Heart Rate PPG Raw Data 200hz
//        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(21), 0);//Heart Rate PPG Raw Data 200hz
//        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(17), 0);//Heart Rate PPG Raw Data 200hz
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(1), 0);//Accelerometer 100hz
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(4), 0);//Gyroscope 100hz
//        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(5), 0);//Gyroscope 100hz
        sensorList = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        Log.e(">>>", "列出当前设备所有的传感器信息");
        for (Sensor sensor : sensorList) {
            Log.e("List sensors", "Name: " + sensor.getName() + " /Type_String: " + sensor.getStringType() + " /Type_number: " + sensor.getType());
        }
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
                    accx.add(Double.valueOf(String.valueOf(e.values[0])));
                    accy.add(Double.valueOf(String.valueOf(e.values[1])));
                    accz.add(Double.valueOf(String.valueOf(e.values[2])));
                    acctimestamps.add(System.currentTimeMillis());
                    break;
                case Sensor.TYPE_GYROSCOPE:     //陀螺传感器
//                    Log.d("Test", "Got the GYROSCOPE : " + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]) + "," + String.valueOf(e.values[2]));

                    gyrx.add(Double.valueOf(String.valueOf(e.values[0])));
                    gyry.add(Double.valueOf(String.valueOf(e.values[1])));
                    gyrz.add(Double.valueOf(String.valueOf(e.values[2])));
                    gyrtimestamps.add(System.currentTimeMillis());
                    break;

                case 65572:   //ppg传感器
//                    Log.d("Test", "raw ppg : "+e.values.length+"," + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[2]));
//                    for(int i=0;i<16;i++){
//                        Log.e(">>>>","e.value:"+i+","+e.values[i]);
//                    }
                    ppgx.add(Double.valueOf(String.valueOf(e.values[0])));
                    //ticwatch的是value1
                    ppgy.add(Double.valueOf(String.valueOf(e.values[1])));
                    ppgtimestamps.add(System.currentTimeMillis());
                    break;
//                case 5:   //光学ppg传感器
//                    //大于100基本就是没在使用
//                    Log.d("Test", "light : " +e.values.length+","+ String.valueOf(e.values[0]));
//                    break;
//                case 17:   //Significant Motion 传感器
//                    //大于100基本就是没在使用
//                    Log.d("Test", "Significant Motion  : " +e.values.length+","+ String.valueOf(e.values[0]));
//                    break;
//                case 21:   // Heart Rate PPG 传感器
//                    //大于100基本就是没在使用
//                    Log.d("Test", " Heart Rate PPG  : " +e.values.length+","+ String.valueOf(e.values[0]));
//                    break;
            }
        }
    };

    public void dataclear() {
        ppgx.clear();
        ppgy.clear();
        ppgtimestamps.clear();

        accx.clear();
        accy.clear();
        accz.clear();
        acctimestamps.clear();

        gyrx.clear();
        gyry.clear();
        gyrz.clear();
        gyrtimestamps.clear();
    }

    public Ppg getnewppgseg(int lens) {
        int length = ppgx.size();
//        Log.e(">>>", "ppgx.size():" + ppgx.size()+"ppgy.size():" + ppgy.size()+"ppgtimestamps.size():" + ppgtimestamps.size());
        if (lens > length - 100) {
            lens = length - 100;
        }
        double[] x = new double[lens];
        double[] y = new double[lens];
        long[] timestamp = new long[lens];

        for (int i = 0; i < lens; i++) {
            x[i] = ppgx.get(i + length - lens - 10);
            y[i] = ppgy.get(i + length - lens - 10);
            timestamp[i] = ppgtimestamps.get(i + length - lens - 10);
        }
        Ppg ppgs = new Ppg(x, y, timestamp);

        return ppgs;
    }

    public Ppg getnewppgseg() {
        return getnewppgseg(ppgx.size());
    }

    public Motion getnewmotionseg(int lens) {
        int length = accx.size();
        if (lens > length - 50) {
            lens = length - 50;
        }
        double[] acc_x = new double[lens];
        double[] acc_y = new double[lens];
        double[] acc_z = new double[lens];
        long[] acctime = new long[lens];

        for (int i = 0; i < lens; i++) {
            acc_x[i] = accx.get(i + length - lens - 10);
            acc_y[i] = accy.get(i + length - lens - 10);
            acc_z[i] = accz.get(i + length - lens - 10);
            acctime[i] = acctimestamps.get(i + length - lens - 10);
        }


        length = gyrx.size();
        if (lens > length - 50) {
            lens = length - 50;
        }
        double[] gyr_x = new double[lens];
        double[] gyr_y = new double[lens];
        double[] gyr_z = new double[lens];
        long[] gyrtime = new long[lens];

        for (int i = 0; i < lens; i++) {
            gyr_x[i] = gyrx.get(i + length - lens - 10);
            gyr_y[i] = gyry.get(i + length - lens - 10);
            gyr_z[i] = gyrz.get(i + length - lens - 10);
            gyrtime[i] = gyrtimestamps.get(i + length - lens - 10);
        }
        Motion motions = new Motion(acc_x,acc_y,acc_z,acctime,gyr_x,gyr_y,gyr_z,gyrtime);

        return motions;
    }



    public void datadelete() {
//        Log.e(">>>","data_delete");
        while (ppgx.size() > 3000) {
            ppgx.remove(0);
            ppgy.remove(0);
            ppgtimestamps.remove(0);
        }

        while (accx.size() > 1500) {
            accx.remove(0);
            accy.remove(0);
            accz.remove(0);
            acctimestamps.remove(0);
        }

        while (gyrx.size() > 1500) {
            gyrx.remove(0);
            gyry.remove(0);
            gyrz.remove(0);
            gyrtimestamps.remove(0);
        }
    }

    public int getppgsize() {
        return ppgx.size();
    }

    public int getaccsize() {
        return accx.size();
    }

    public int getgyrsize() {
        return gyrx.size();
    }
}
