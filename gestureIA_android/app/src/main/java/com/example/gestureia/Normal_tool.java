package com.example.gestureia;

import android.util.Log;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;

import uk.me.berndporr.iirj.Butterworth;

public class Normal_tool {
    public Variance vars = new Variance();
    public Mean means = new Mean();
    public Max maxs = new Max();
    public Min mins = new Min();
    public StandardDeviation stds = new StandardDeviation();

    public double KL_divergence(double[] P, double[] Q) {
        double KL = 0;
        for (int i = 0; i < P.length; i++) {
            KL = KL + P[i] * Math.log(P[i] / Q[i]);
        }
        return KL;
    }

    public double JS_divergence(double[] P, double[] Q) {
        double JS = 0;
        double[] M = new double[P.length];
        double[] _P = normalscale(P);
        double[] _Q = normalscale(Q);

        for (int i = 0; i < P.length; i++) {
            M[i] = (_P[i] + _Q[i]) / 2;
        }
        JS = (KL_divergence(_P, M) + KL_divergence(_Q, M)) / 2;
        return JS;
    }

    // 相关系数
    public double cal_corr(double[] x, double[] y) {
        double x_mean = means.evaluate(x);
        double y_mean = means.evaluate(y);
        double cov_xy = 0;
        double sqx = 0;
        double sqy = 0;
        for (int i = 0; i < x.length; i++) {
            cov_xy += (x[i] - x_mean) * (y[i] - y_mean);
            sqx += (x[i] - x_mean) * (x[i] - x_mean);
            sqy += (y[i] - y_mean) * (y[i] - y_mean);
        }

        double sq = Math.sqrt(sqx * sqy);
        // 中间的N上下约去
        double cor = cov_xy / sq;
//		System.out.println("cov:" + cov_xy + ",sq:" + sq + ",cor:" + cor);
        return cor;
    }

    // 获取自回归系数
    public double get_auto_corr(double[] data, int k) {
        int datalen = data.length;
        double[] temp1 = new double[datalen - k];
        double[] temp2 = new double[datalen - k];
        for (int i = 0; i < datalen - k; i++) {
            temp1[i] = data[i];
            temp2[i] = data[k + i];
        }
        double data_var = vars.evaluate(data) * datalen;
        double data_mean = means.evaluate(data);
        double auto_corr = 0;
        for (int i = 0; i < datalen - k; i++) {
            auto_corr += (temp1[i] - data_mean) * (temp2[i] - data_mean) / data_var;
        }
        return auto_corr;
    }

    // 按照间隙，进行均值滤波
    public double[] meanfilt(double[] data, int interval) {
        int datasize = data.length;
        double[] templist = new double[datasize - interval];
        for (int i = 0; i < datasize - interval; i++) {
            templist[i] = means.evaluate(data, i, interval);
        }
        return templist;
    }

    // 0-1标准化
    public double[] minmaxscale(double[] data) {
        double[] templist = new double[data.length];
        double tempmax = maxs.evaluate(data);
        double tempmin = mins.evaluate(data);
        double interval = tempmax - tempmin;
        for (int i = 0; i < data.length; i++) {
            templist[i] = (data[i] - tempmin) / interval;
        }
        return templist;
    }

    // z-score标准化
    public double[] standardscale(double[] data) {
        double[] templist = new double[data.length];
        double datamean = means.evaluate(data);
        double datavar = stds.evaluate(data);
        for (int i = 0; i < data.length; i++) {
            templist[i] = (data[i] - datamean) / datavar;
        }
        return templist;
    }

    // 归一化，范数1
    public double[] normalscale(double[] data) {
        double[] templist = new double[data.length];
        double norm = 0;
        for (int i = 0; i < data.length; i++) {
            norm += data[i];
        }
        for (int i = 0; i < data.length; i++) {
            templist[i] = data[i] / norm;
        }
        return templist;
    }

    // 初始的butterworth滤波会有个收敛的过程，前面的数据手动选择滤除
    public double[] butterworth_highpass(double[] data, int fre, int high) {
        double[] templist = new double[data.length];
        Butterworth butterworthx = new Butterworth();
        butterworthx.highPass(3, fre, high);// order,fre,cutoff
        for (int i = 0; i < data.length; i++) {
            templist[i] = butterworthx.filter(data[i]);
        }
        return templist;
    }

    public double[] butterworth_lowpass(double[] data, int fre, int low) {
        double[] templist = new double[data.length];
        Butterworth butterworthx = new Butterworth();
        butterworthx.highPass(3, fre, low);// order,fre,cutoff
        for (int i = 0; i < data.length; i++) {
            templist[i] = butterworthx.filter(data[i]);
        }
        return templist;
    }

    public double[] butterworth_bandpass(double[] data, int fre, int high, int low) {
        int center = ((high + low) / 2);
        int width = high - low;
        double[] templist = new double[data.length];
        Butterworth butterworthx = new Butterworth();
        butterworthx.highPass(3, fre, center, width);// order,fre,center,width
        for (int i = 0; i < data.length; i++) {
            templist[i] = butterworthx.filter(data[i]);
        }
        return templist;
    }

    // 序列转矩阵
    public double[] arraytomatrix(ArrayList<Double> data) {
        ArrayList<Double> finaldata=new ArrayList<Double>();

        for (int i = 0; i < data.size(); i++) {
                if(data.get(i)!=null){
                    finaldata.add(data.get(i));
                }
        }
        data=finaldata;
        int datalength = data.size();

        double[] matrix = new double[datalength];
        for (int i = 0; i < datalength; i++) {
            matrix[i] = data.get(i);
        }
        return matrix;
    }

    // 矩阵转序列
    public ArrayList<Double> matrixtoarray(double[] data) {
        int datalength = data.length;
        ArrayList<Double> array = new ArrayList<Double>();
        for (int i = 0; i < datalength; i++) {
            array.add(data[i]);
        }
        return array;
    }

    public double[] array_dataselect(double[] data, int start, int lens) {
        double[] templist = new double[lens];
        for (int i = 0; i <lens; i++) {
            templist[i] = data[i +start];
        }
        return templist;
    }
    public double[] strarraytodoublearray(String[] str){
        double[] lo=new double[str.length];
        for(int i=0;i<str.length;i++){
            lo[i]=Double.parseDouble(str[i]);
        }
        return lo;
    }

    public int[] strarraytointarray(String[] str){
        int[] lo=new int[str.length];
        for(int i=0;i<str.length;i++){
            lo[i]=Integer.parseInt(str[i]);
        }
        return lo;
    }

}
