package com.example.gestureia;

public class Motion {

    public double[] accx = null;
    public double[] accy = null;
    public double[] accz = null;
    public long[] acctimestamps = null;

    public double[] gyrx = null;
    public double[] gyry = null;
    public double[] gyrz = null;
    public long[] gyrtimestamps = null;

    public Motion(double[] acc_x, double[] acc_y, double[] acc_z,long[] acctime,
                  double[] gyr_x, double[] gyr_y, double[] gyr_z,long[] gyrtime) {
        accx = acc_x;
        accy = acc_y;
        accz = acc_z;
        acctimestamps=acctime;

        gyrx = gyr_x;
        gyry = gyr_y;
        gyrz = gyr_z;
        gyrtimestamps = gyrtime;
    }
    public Motion(){}
}