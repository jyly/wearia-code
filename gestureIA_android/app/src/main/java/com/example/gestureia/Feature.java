package com.example.gestureia;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import org.apache.commons.math3.stat.descriptive.moment.Skewness;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.complex.Complex;

import jwave.Transform;
import jwave.transforms.FastWaveletTransform;
import jwave.transforms.wavelets.haar.Haar1;

public class Feature {
	public ArrayList<Double> features = new ArrayList<Double>();

	public StandardDeviation stds = new StandardDeviation();
	public Mean means = new Mean();
	public Max maxs = new Max();
	public Min mins = new Min();
	public Kurtosis kurtosis = new Kurtosis();
	public Skewness skewness = new Skewness();
	public Percentile percentile = new Percentile();

	public IAtool iatools = new IAtool();
	public Normal_tool nortools = new Normal_tool();

	FastFourierTransformer ffts = new FastFourierTransformer(DftNormalization.STANDARD);

	public void ppg_feature(double[] data) {

		double[] tempdata = iatools.to2power(data);
		int datalen = data.length;

		features.add(means.evaluate(data));
		features.add(stds.evaluate(data));
		features.add(maxs.evaluate(data) - mins.evaluate(data));
		features.add(maxs.evaluate(data));
		features.add(mins.evaluate(data));
		features.add(percentile.evaluate(data, 50));
		features.add(kurtosis.evaluate(data));
		features.add(skewness.evaluate(data));

		double mean = means.evaluate(data);
		double rms = 0, absamplitude = 0, diversion = 0;
		for (int i = 0; i < datalen; i++) {
			rms = rms + data[i] * data[i];
			absamplitude = absamplitude + Math.abs(data[i] - mean);
			diversion = diversion + Math.abs(data[i]);
		}
		features.add(Math.sqrt(rms / datalen));
		features.add(absamplitude / datalen);
		features.add(diversion / datalen);

		double[] interval = iatools.interationcal(data);
		features.add(maxs.evaluate(interval));
		features.add(mins.evaluate(interval));
		features.add(kurtosis.evaluate(interval));
		features.add(skewness.evaluate(interval));
		features.add(percentile.evaluate(interval, 50));

		rms = 0;
		absamplitude = 0;
		diversion = 0;
		mean = means.evaluate(interval);
		for (int i = 0; i < datalen - 1; i++) {
			rms = rms + interval[i] * interval[i];
			absamplitude = absamplitude + Math.abs(interval[i] - mean);
			diversion = diversion + Math.abs(interval[i]);
		}
		features.add(Math.sqrt(rms / (datalen - 1)));
		features.add(absamplitude / (datalen - 1));
		features.add(diversion / (datalen - 1));

		Complex[] datacomplex = ffts.transform(tempdata, TransformType.FORWARD);
		Fftvalue tempvalue = iatools.fftcal(datacomplex, 200);
		ArrayList<Double> extrafft = new ArrayList<Double>();
		for (int i = 0; i < tempvalue.fftscore.size(); i++) {
			if (tempvalue.fluency.get(i) < 5) {
				extrafft.add(tempvalue.fftscore.get(i));
			} else {
				break;
			}
		}
		double[] extrafftmatrix = nortools.arraytomatrix(extrafft);
		features.add(means.evaluate(extrafftmatrix));
		features.add(stds.evaluate(extrafftmatrix));
		features.add(maxs.evaluate(extrafftmatrix) - mins.evaluate(extrafftmatrix));
		features.add(maxs.evaluate(extrafftmatrix));
		features.add(mins.evaluate(extrafftmatrix));
		features.add(percentile.evaluate(extrafftmatrix, 50));
		features.add(kurtosis.evaluate(extrafftmatrix));
		features.add(skewness.evaluate(extrafftmatrix));

		// 分解成9阶的离散小波
		double[][] coeffs = null;
		Transform t = new Transform(new FastWaveletTransform(new Haar1()));
		coeffs = t.decompose(tempdata);
//		for (int i = 0; i < coeffs.length; i++) {
//			for (int j = 0; j < coeffs[i].length; j++) {
//				System.out.print("coeffs[" + i + "][" + j + "]��" + coeffs[i][j] + ",");
//			}
//			System.out.println(" ");
//		}
		double[] selecteffs = coeffs[3];
		features.add(means.evaluate(selecteffs));
		features.add(stds.evaluate(selecteffs));
		features.add(maxs.evaluate(selecteffs) - mins.evaluate(selecteffs));

		rms = 0;
		absamplitude = 0;
		diversion = 0;
		mean = means.evaluate(selecteffs);
		for (int i = 0; i < selecteffs.length; i++) {
			rms = rms + selecteffs[i] * selecteffs[i];
			absamplitude = absamplitude + Math.abs(selecteffs[i] - mean);
			diversion = diversion + Math.abs(selecteffs[i]);
		}
		features.add(Math.sqrt(rms / selecteffs.length));
		features.add(absamplitude / selecteffs.length);
		features.add(diversion / selecteffs.length);

		// 自回归系数
		for (int i = 1; i < 10; i++) {
			features.add(nortools.get_auto_corr(data, i));
		}
//				for(int i=0;i<features.size();i++) {
//					System.out.println("feature["+i+"]:"+features.get(i));
//				}

	}
}
