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
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(1), 0);//Accelerometer 100hz
        mSensorManager.registerListener(listener, mSensorManager.getDefaultSensor(4), 0);//Gyroscope 100hz
        sensorList = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        Log.e(">>>","列出当前设备所有的传感器信息");
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
//                    Log.d("Test", "raw ppg : " + String.valueOf(e.values[0]) + "," + String.valueOf(e.values[1]) + "," + timeStamp);
                    ppgx.add(Double.valueOf(String.valueOf(e.values[0])));
                    ppgy.add(Double.valueOf(String.valueOf(e.values[1])));
                    ppgtimestamps.add(System.currentTimeMillis());
                    break;
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
        Ppg ppgs = new Ppg();
        int length = ppgx.size();
//        Log.e(">>>", "ppgx.size():" + ppgx.size()+"ppgy.size():" + ppgy.size()+"ppgtimestamps.size():" + ppgtimestamps.size());

        if (lens > length - 100) {
            lens = length - 100;
        }
        for (int i = length - lens; i < length - 10; i++) {
            ppgs.x.add(ppgx.get(i));
            ppgs.y.add(ppgy.get(i));
            ppgs.timestamps.add(ppgtimestamps.get(i));
        }
        return ppgs;
    }

    public Ppg getnewppgseg() {
        return getnewppgseg(ppgx.size());
    }

    public Motion getnewmotionseg(int lens) {
        Motion motions = new Motion();
        int length = accx.size();
        if (lens > length - 50) {
            lens = length - 50;
        }
//        Log.e(">>>", "accx.size:" + accx.size()+"accy.size:" + accy.size()+"accz.size:" + accz.size()+"acctimestamps.size:" + acctimestamps.size());
        for (int i = length - lens; i < length - 10; i++) {
            motions.accx.add(accx.get(i));
            motions.accy.add(accy.get(i));
            motions.accz.add(accz.get(i));
            motions.acctimestamps.add(acctimestamps.get(i));
        }

        length = gyrx.size();
        if (lens > length - 50) {
            lens = length - 50;
        }
//        Log.e(">>>", "gyrx.size:" + gyrx.size()+"gyry.size:" + gyry.size()+"gyrz.size:" + gyrz.size()+"gyrtimestamps.size:" + gyrtimestamps.size());
        for (int i = length - lens; i < length - 10; i++) {
            motions.gyrx.add(gyrx.get(i));
            motions.gyry.add(gyry.get(i));
            motions.gyrz.add(gyrz.get(i));
            motions.gyrtimestamps.add(gyrtimestamps.get(i));
        }
        return motions;
    }


    public Motion getnewmotionseg(int acclens, int gyrlens) {
        Motion motions = new Motion();
        int length = accx.size();
        if (acclens > length - 30) {
            acclens = length - 30;
        }
//        Log.e(">>>", "accx.size:" + accx.size()+"accy.size:" + accy.size()+"accz.size:" + accz.size()+"acctimestamps.size:" + acctimestamps.size());
        for (int i = length - acclens; i < length - 10; i++) {
            motions.accx.add(accx.get(i));
            motions.accy.add(accy.get(i));
            motions.accz.add(accz.get(i));
            motions.acctimestamps.add(acctimestamps.get(i));
        }

        length = gyrx.size();
        if (gyrlens > length - 30) {
            gyrlens = length - 30;
        }
//        Log.e(">>>", "gyrx.size:" + gyrx.size()+"gyry.size:" + gyry.size()+"gyrz.size:" + gyrz.size()+"gyrtimestamps.size:" + gyrtimestamps.size());
        for (int i = length - gyrlens; i < length - 10; i++) {
            motions.gyrx.add(gyrx.get(i));
            motions.gyry.add(gyry.get(i));
            motions.gyrz.add(gyrz.get(i));
            motions.gyrtimestamps.add(gyrtimestamps.get(i));
        }
        return motions;
    }

    public Motion getnewmotionseg() {
        return getnewmotionseg(accx.size(), gyrx.size());
    }

    public void datadelete() {
//        Log.e(">>>","data_delete");
        while(ppgx.size()>3000){
            ppgx.remove(0);
            ppgy.remove(0);
            ppgtimestamps.remove(0);
        }

        while(accx.size()>1500){
            accx.remove(0);
            accy.remove(0);
            accz.remove(0);
            acctimestamps.remove(0);
        }

        while(gyrx.size()>1500){
            gyrx.remove(0);
            gyry.remove(0);
            gyrz.remove(0);
            gyrtimestamps.remove(0);
        }
    }

    public int getppgsize(){
        return  ppgx.size();
    }

    public int getaccsize(){
        return  accx.size();
    }
    public int getgyrsize(){
        return  gyrx.size();
    }
}
