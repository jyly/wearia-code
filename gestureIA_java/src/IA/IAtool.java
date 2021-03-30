package IA;

import java.util.ArrayList;
import org.fastica.FastICA;
import org.fastica.FastICAException;

import org.apache.commons.math3.stat.descriptive.moment.*;
import org.apache.commons.math3.complex.Complex;

public class IAtool {

	public Normal_tool nortools = new Normal_tool();

	// 将ppg的array转化成矩阵，混合信号矩阵
	public double[][] constructmixsignal(PPG ppgs) {
		int arraylength = ppgs.x.length;
		double[][] mixedSignal = new double[2][arraylength];
		mixedSignal[0] = ppgs.x;
		mixedSignal[1] = ppgs.y;
		return mixedSignal;
	}

	// 将分离后的信号矩阵恢复为ppg
	public PPG constructnewppg(double[] x, double[] y) {
		PPG ppgs = new PPG();
		ppgs.x = x;
		ppgs.y = y;
		return ppgs;
	}

	// 快速独立成分分析
	public PPG fastica(PPG ppgs) {
		int arraylength = ppgs.x.length;

		double[][] mixedSignal = constructmixsignal(ppgs);
		double[][] cleanSignal = new double[2][arraylength];
		try {
			FastICA fica = new FastICA(mixedSignal, 2);
			cleanSignal = fica.getICVectors();
		} catch (FastICAException e) {
			e.printStackTrace();
		}

		PPG temp = ppgs;
		if (2 == cleanSignal.length) {
			temp = constructnewppg(cleanSignal[0], cleanSignal[1]);
		}

		return temp;
	}

	// 根据峰值判断那条手势信号和脉冲信号
	public PPG machoice(PPG ppgs) {
		double[][] tempSignal = constructmixsignal(ppgs);
		Kurtosis kurtosis = new Kurtosis();
		double xkur = kurtosis.evaluate(tempSignal[0]);
		double ykur = kurtosis.evaluate(tempSignal[1]);
		System.out.println("xkur:" + xkur + " ykur:" + ykur);
		PPG temp = new PPG();
		if (Math.abs(xkur) > (Math.abs(ykur))) {
			temp = constructnewppg(tempSignal[0], tempSignal[1]);
		} else {
			temp = constructnewppg(tempSignal[1], tempSignal[0]);
		}
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
			} else if (datalen < 1024) {
				flag = 1024;
			} else {
				flag = 2048;
			}
		}
		for (int i = datalen; i < flag; i++) {
			tempmatrix.add((double) 0);
		}
		double[] matrix = nortools.arraytomatrix(tempmatrix);
		// System.out.println(matrix.length);
		// for (int i = 0; i < matrix.length; i++) {
		// System.out.println(matrix[i]);
		// }
		return matrix;
	}

	// 计算傅里叶变换的振幅和对应的频率
	public FFTvalue fftcal(Complex[] x, double fre) {
		int xlen = (int) (x.length / 2);

		double[] fftscore = new double[xlen];
		double[] fluency = new double[xlen];
		for (int i = 0; i < xlen; i++) {
			fftscore[i] = (Math.sqrt(x[i].getReal() * x[i].getReal() + x[i].getImaginary() * x[i].getImaginary())
					/ (double) (x.length));
			fluency[i] = (i / (double) (x.length) * fre);
		}
		FFTvalue tempvalue = new FFTvalue(fluency, fftscore);

		// for (int i = 0; i < xlen; i++) {
		// System.out.println("fft" + i + "=" + fftscore.get(i) + " fluency" + i + "=" +
		// fluency.get(i));
		// }
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
		return nortools.arraytomatrix(templist);
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
		// Log.e(">>>","tag.length"+tag.length);
		// for (int i=0;i<tag.length;i++){
		// Log.e("PQtag","第"+i+"个pg:"+Ptag[i]+","+Qtag[i]);
		// }

		for (int i = 0; i < tag.length; i++) {
			if (0 == Ptag[i]) {
				Ptag[i] = 0.00000001;

			}
			if (0 == Qtag[i]) {
				Qtag[i] = 0.00000001;
			}
			// Log.e("PQtag","第"+i+"个pg:"+Ptag[i]+","+Qtag[i]);
		}
		// return tools.KL_divergence(Ptag,Qtag);
		return nortools.JS_divergence(Ptag, Qtag);
	}

	public float[] featurestd(float feature[], double[] scale_mean, double[] scale_scale) {
		float[] finalfeature = new float[feature.length];
		for (int i = 0; i < feature.length; i++) {
			finalfeature[i] = (feature[i] - (float) scale_mean[i]) / (float) scale_scale[i];
		}
		return finalfeature;
	}
}
