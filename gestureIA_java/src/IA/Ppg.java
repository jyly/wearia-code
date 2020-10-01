package IA;


public class Ppg {
    public double[] x = null;
    public double[] y = null;
    public long[] timestamps = null;

    public Ppg(double[] ppg_x, double[] ppg_y,long[] ppgtime) {
        x = ppg_x;
        y = ppg_y;
        timestamps=ppgtime;
    }
    public Ppg() {}
    public Ppg(int lens) {
        x = new double[lens];
        y = new double[lens];
        timestamps=new long [lens];
    }
}
