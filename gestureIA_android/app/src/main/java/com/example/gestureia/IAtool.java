package com.example.gestureia;


import android.util.Log;

import org.fastica.FastICA;
import org.fastica.FastICAException;
import java.util.ArrayList;
import org.apache.commons.math3.stat.descriptive.moment.*;
import org.apache.commons.math3.complex.Complex;

public class IAtool {

    // 将ppg的array转化成矩阵，混合信号矩阵
    public double[][] constructmixsignal(Ppg ppgs) {
        int arraylength = ppgs.x.length;
        double[][] mixedSignal = new double[2][arraylength];
        mixedSignal[0] = ppgs.x;
        mixedSignal[1] = ppgs.y;
        return mixedSignal;
    }

    // 将分离后的信号矩阵恢复为ppg
    public Ppg constructnewppg(double[] x, double[] y) {
        Ppg ppgs = new Ppg();
        ppgs.x=x;
        ppgs.y=y;
        return ppgs;
    }

    // 快速独立成分分析
    public Ppg fastica(Ppg ppgs) {
        int arraylength = ppgs.x.length;
        double[][] mixedSignal = constructmixsignal(ppgs);
        double[][] cleanSignal = new double[2][arraylength];
        try {
            FastICA fica = new FastICA(mixedSignal, 2);
            cleanSignal = fica.getICVectors();
        } catch (FastICAException e) {
            e.printStackTrace();
        }
        Ppg temp = constructnewppg(cleanSignal[0], cleanSignal[1]);
        mixedSignal=null;
        cleanSignal=null;
        return temp;
    }

    // 根据峰值判断那条手势信号和脉冲信号
    public Ppg machoice(Ppg ppgs) {
        double[][] tempSignal = constructmixsignal(ppgs);
        Kurtosis kurtosis = new Kurtosis();
        double xkur = kurtosis.evaluate(tempSignal[0]);
        double ykur = kurtosis.evaluate(tempSignal[1]);
        kurtosis=null;
        System.out.println("xkur:" + xkur + " ykur:" + ykur);
        Ppg temp = new Ppg();
        if (Math.abs(xkur) > (Math.abs(ykur))) {
            temp = constructnewppg(tempSignal[0], tempSignal[1]);
        } else {
            temp = constructnewppg(tempSignal[1], tempSignal[0]);
        }
        tempSignal=null;
        return temp;
    }

    // 将在不是2的幂次方的序列后面补零到2的幂次方
    public double[] to2power(double[] data) {
        ArrayList<Double> tempmatrix = new ArrayList<Double>();
        int datalen = data.length;

        for (int i = 0; i < datalen; i++) {
            tempmatrix.add(data[i]);
        }
        int flag = 0;
        if (datalen < 256) {
            flag = 256;

        } else {
            if (datalen < 512) {
                flag = 512;
            } else {
                flag = 1024;
            }
        }
        for (int i = datalen; i < flag; i++) {
            tempmatrix.add((double)0);
        }
       Normal_tool nortools = new Normal_tool();
        double[] matrix = nortools.arraytomatrix(tempmatrix);
        nortools=null;
        tempmatrix=null;
//		System.out.println(matrix.length);
//		for (int i = 0; i < matrix.length; i++) {
//			System.out.println(matrix[i]);
//		}
        return matrix;
    }

    // 计算傅里叶变换的振幅和对应的频率
    public Fftvalue fftcal(Complex[] x, double fre) {
        int xlen = (int) (x.length / 2);

        double [] fftscore=new double[xlen];
        double [] fluency=new double[xlen];
        for (int i = 0; i < xlen; i++) {
            fftscore[i]=(Math.sqrt(x[i].getReal() * x[i].getReal() + x[i].getImaginary() * x[i].getImaginary())
                            / (double) (x.length));
            fluency[i]=(i / (double) (x.length) * fre);
        }
        Fftvalue tempvalue = new Fftvalue(fluency,fftscore);
        fluency=null;
        fftscore=null;
//		for (int i = 0; i < xlen; i++) {
//			System.out.println("fft" + i + "=" + fftscore.get(i) + "   fluency" + i + "=" + fluency.get(i));
//		}
        return tempvalue;
    }

    public double[] interationcal(double[] data) {
        double[] interval = new double[data.length - 1];
        for (int i = 0; i < data.length - 1; i++) {
            interval[i] = data[i + 1] - data[i];
        }
        return interval;
    }

    public double[] tagcal(double[] data) {
        ArrayList<Double> templist = new ArrayList<Double>();
        for (int i = 0; i < data.length; i++) {
            if (!templist.contains(data[i])) {
                templist.add(data[i]);
            }
        }
        Normal_tool nortools = new Normal_tool();
        double[] tag=nortools.arraytomatrix(templist);
        nortools=null;
        return tag;
    }

    public double array_JS_cal(double[] P, double[] Q, double[] tag) {
        double[] Ptag = new double[tag.length];
        double[] Qtag = new double[tag.length];
        for (int i = 0; i < tag.length; i++) {
            Ptag[i] = 0;
            Qtag[i] = 0;
        }
        for (int i = 0; i < P.length; i++) {
            for (int j = 0; j < tag.length; j++) {
                if (P[i] == tag[j]) {
                    Ptag[j] += 1;
                    continue;
                }
            }
        }
        for (int i = 0; i < Q.length; i++) {
            for (int j = 0; j < tag.length; j++) {
                if (Q[i] == tag[j]) {
                    Qtag[j] += 1;
                    continue;
                }
            }
        }
//	        Log.e(">>>","tag.length"+tag.length);
//	        for (int i=0;i<tag.length;i++){
//	           Log.e("PQtag","第"+i+"个pg:"+Ptag[i]+","+Qtag[i]);
//	        }

        for (int i = 0; i < tag.length; i++) {
            if (0==Ptag[i]) {
                Ptag[i] = 0.00000001;

            }
            if (0==Qtag[i]) {
                Qtag[i] = 0.00000001;
            }

        }

        Normal_tool nortools = new Normal_tool();
        double score=nortools.JS_divergence(Ptag, Qtag);
        Ptag=null;
        Qtag=null;
        nortools=null;
        return score;
    }



}
