package IA;


public class PPG {
    public double[] x = null;
    public double[] y = null;
    public long[] timestamps = null;

    public PPG(double[] ppg_x, double[] ppg_y,long[] ppgtime) {
        x = ppg_x;
        y = ppg_y;
        timestamps=ppgtime;
    }
    public PPG() {}
    public PPG(int lens) {
        x = new double[lens];
        y = new double[lens];
        timestamps=new long [lens];
    }
}
