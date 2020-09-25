package IA;

import java.util.ArrayList;
import org.fastica.FastICA;
import org.fastica.FastICAException;

import org.apache.commons.math3.stat.descriptive.moment.*;
import org.apache.commons.math3.complex.Complex;

import uk.me.berndporr.iirj.Butterworth;

public class IAtool {

	public normal_tool nortools = new normal_tool();

	public double[] dataselect(double[] data, int start, int lens) {
		double[] templist = new double[lens];
		for (int i = start; i < start + lens; i++) {
			templist[i] = data[i];
		}
		return templist;
	}

	// ��ppg��arrayת���ɾ��󣬻���źž���
	public double[][] constructmixsignal(ppg ppgs) {
		int arraylength = ppgs.x.size();
		double[][] mixedSignal = new double[2][arraylength];
		for (int i = 0; i < arraylength; i++) {
			mixedSignal[0][i] = ppgs.x.get(i);
			mixedSignal[1][i] = ppgs.y.get(i);
		}
		return mixedSignal;
	}

	// ���������źž���ָ�Ϊppg
	public ppg constructnewppg(double[] x, double[] y) {
		int arraylength = x.length;
		ppg ppgs = new ppg();
		for (int i = 0; i < arraylength; i++) {
			ppgs.x.add(x[i]);
			ppgs.y.add(y[i]);
		}
		return ppgs;
	}

	// ���ٶ����ɷַ���
	public ppg fastica(ppg ppgs) {
		int arraylength = ppgs.x.size();
		double[][] mixedSignal = constructmixsignal(ppgs);
		double[][] cleanSignal = new double[2][arraylength];

		try {
			FastICA fica = new FastICA(mixedSignal, 2);
			cleanSignal = fica.getICVectors();
		} catch (FastICAException e) {
			e.printStackTrace();
		}

		ppg temp = constructnewppg(cleanSignal[0], cleanSignal[1]);
		return temp;
	}

	// ���ݷ�ֵ�ж����������źź������ź�
	public ppg machoice(ppg ppgs) {
		double[][] tempSignal = constructmixsignal(ppgs);
		Kurtosis kurtosis = new Kurtosis();
		double xkur = kurtosis.evaluate(tempSignal[0]);
		double ykur = kurtosis.evaluate(tempSignal[1]);
//		System.out.println("xkur:" + xkur + " ykur:" + ykur);
		ppg temp = new ppg();
		if (Math.abs(xkur) > (Math.abs(ykur))) {
			temp = constructnewppg(tempSignal[0], tempSignal[1]);
		} else {
			temp = constructnewppg(tempSignal[1], tempSignal[0]);
		}
		return temp;
	}

	// ���ڲ���2���ݴη������к��油�㵽2���ݴη�
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
			tempmatrix.add((double) 0);
		}
		double[] matrix = nortools.arraytomatrix(tempmatrix);
//		System.out.println(matrix.length);
//		for (int i = 0; i < matrix.length; i++) {
//			System.out.println(matrix[i]);
//		}
		return matrix;
	}

	// ���㸵��Ҷ�任������Ͷ�Ӧ��Ƶ��
	public fftvalue fftcal(Complex[] x, double fre) {
		fftvalue tempvalue = new fftvalue();

		int xlen = (int) (x.length / 2);
		for (int i = 0; i < xlen; i++) {
			tempvalue.fftscore
					.add(Math.sqrt(x[i].getReal() * x[i].getReal() + x[i].getImaginary() * x[i].getImaginary())
							/ (double) (x.length));
			tempvalue.fluency.add(i / (double) (x.length) * fre);
		}

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
//	        Log.e(">>>","tag.length"+tag.length);
//	        for (int i=0;i<tag.length;i++){
//	           Log.e("PQtag","��"+i+"��pg:"+Ptag[i]+","+Qtag[i]);
//	        }

		for (int i = 0; i < tag.length; i++) {
			if (Ptag[i] == 0) {
				Ptag[i] = 0.00000001;

			} 
			if (Qtag[i] == 0) {
				Qtag[i] = 0.00000001;
			} 
//	            Log.e("PQtag","��"+i+"��pg:"+Ptag[i]+","+Qtag[i]);
		}
//	        return tools.KL_divergence(Ptag,Qtag);
		return nortools.JS_divergence(Ptag, Qtag);
	}

}
