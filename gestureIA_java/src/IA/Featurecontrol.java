package IA;



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

public class Featurecontrol {
	public double[] based_feature(double[] data) {
		Normal_tool nortools = new Normal_tool();
		StandardDeviation stds = new StandardDeviation();
		Mean means = new Mean();
		Max maxs = new Max();
		Min mins = new Min();
		Kurtosis kurtosis = new Kurtosis();
		Skewness skewness = new Skewness();
		Percentile percentile = new Percentile();
		FastFourierTransformer ffts = new FastFourierTransformer(DftNormalization.STANDARD);

		ArrayList<Double> features = new ArrayList<Double>();
		 IAtool iatools = new IAtool();

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

		double rms = 0, diversion = 0;
		for (int i = 0; i < datalen; i++) {
			rms = rms + data[i] * data[i];
			diversion = diversion + Math.abs(data[i]);
		}

		features.add(Math.sqrt(rms / datalen));
		features.add(diversion / datalen);

		double[] interval = iatools.interationcal(data);

		features.add(maxs.evaluate(interval));
		features.add(mins.evaluate(interval));
		features.add(kurtosis.evaluate(interval));
		features.add(skewness.evaluate(interval));

		rms = 0;
		diversion = 0;

		for (int i = 0; i < datalen - 1; i++) {
			rms = rms + interval[i] * interval[i];
			diversion = diversion + Math.abs(interval[i]);
		}
		interval=null;
		features.add(Math.sqrt(rms / (datalen - 1)));
		features.add(diversion / (datalen - 1));

		Complex[] datacomplex = ffts.transform(tempdata, TransformType.FORWARD);
		Fftvalue tempvalue = iatools.fftcal(datacomplex, 200);
		datacomplex=null;

		ArrayList<Double> extrafft = new ArrayList<Double>();
		for (int i = 0; i < tempvalue.fftscore.length; i++) {
			if (tempvalue.fluency[i] < 5) {
				extrafft.add(tempvalue.fftscore[i]);
			} else {
				break;
			}
		}
		tempvalue=null;
		double[] extrafftmatrix = nortools.arraytomatrix(extrafft);
		extrafft=null;
		features.add(means.evaluate(extrafftmatrix));
		features.add(stds.evaluate(extrafftmatrix));
		features.add(maxs.evaluate(extrafftmatrix) - mins.evaluate(extrafftmatrix));
		features.add(maxs.evaluate(extrafftmatrix));
		features.add(mins.evaluate(extrafftmatrix));
		features.add(percentile.evaluate(extrafftmatrix, 50));
		features.add(kurtosis.evaluate(extrafftmatrix));
		features.add(skewness.evaluate(extrafftmatrix));
		extrafftmatrix=null;

		// 分解成9阶的离散小波
		Transform t = new Transform(new FastWaveletTransform(new Haar1()));
		double[][] coeffs = t.decompose(tempdata);
//		for (int i = 0; i < coeffs.length; i++) {
//			for (int j = 0; j < coeffs[i].length; j++) {
//				System.out.print("coeffs[" + i + "][" + j + "]��" + coeffs[i][j] + ",");
//			}
//			System.out.println(" ");
//		}
		double[] selecteffs = coeffs[3];
		t=null;
		coeffs=null;

		features.add(means.evaluate(selecteffs));
		features.add(stds.evaluate(selecteffs));
		features.add(maxs.evaluate(selecteffs) - mins.evaluate(selecteffs));

		rms = 0;
		diversion = 0;
		for (int i = 0; i < selecteffs.length; i++) {
			rms = rms + selecteffs[i] * selecteffs[i];
			diversion = diversion + Math.abs(selecteffs[i]);
		}
		features.add(Math.sqrt(rms / selecteffs.length));
		features.add(diversion / selecteffs.length);
		selecteffs=null;

		// 自回归系数
		for (int i = 1; i < 10; i++) {
			features.add(nortools.get_auto_corr(data, i));
		}
//				for(int i=0;i<features.size();i++) {
//					System.out.println("feature["+i+"]:"+features.get(i));
//				}

		double [] finalfeature=nortools.arraytomatrix(features);

		features=null;
		tempdata=null;
		stds = null;
		means =null;
		maxs = null;
		mins = null;
		kurtosis = null;
		skewness = null;
		percentile =null;
		ffts = null;
		iatools=null;
		nortools=null;
		return finalfeature;
	}



    public double[] return_feature(Ppg ppgs, Motion motion, Ppg butterppg, Ppg icappg) {
		Normal_tool nortools = new Normal_tool();
		ArrayList<Double> samplefeature = new ArrayList<Double>();
		double[] finalfeature = nortools.arraytomatrix(samplefeature);
		nortools = null;
		return finalfeature;

	}

    public double[] return_feature(Ppg ppgs, Motion motion) {
        ArrayList<Double> samplefeature = new ArrayList<Double>();
        double[] temp = based_feature(ppgs.x);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(ppgs.y);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.accx);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.accy);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.accz);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.gyrx);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.gyry);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(motion.gyrz);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        Normal_tool nortools = new Normal_tool();
        double[] finalfeature = nortools.arraytomatrix(samplefeature);
        temp = null;
        samplefeature = null;
        nortools = null;
        return finalfeature;
    }

    public double[] return_feature(Ppg ppgs) {
        ArrayList<Double> samplefeature = new ArrayList<Double>();
        double[] temp = based_feature(ppgs.x);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        temp = based_feature(ppgs.y);
        for (int i = 0; i < temp.length; i++) {
            samplefeature.add(temp[i]);
        }
        Normal_tool nortools = new Normal_tool();
        double[] finalfeature = nortools.arraytomatrix(samplefeature);
        temp = null;
        samplefeature = null;
        nortools = null;
        return finalfeature;
    }

    public double[] build_feature(Ppg ppgs, Motion motions) {
        MAfind ma = new MAfind();
        Normal_tool nortools = new Normal_tool();
        IAtool iatools = new IAtool();

        double[] samplefeature = null;

        Ppg orippg = new Ppg();
        orippg.x = nortools.meanfilt(ppgs.x, 20);
        orippg.y = nortools.meanfilt(ppgs.y, 20);

        Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
        butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
        butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);
        // 做快速主成分分析
        Ppg icappg = iatools.fastica(butterppg);
        // 根据峰值判断那条手势信号和脉冲信号
        icappg = iatools.machoice(icappg);
        //细粒度手势分析，判断手势区间
//		int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
        int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5,0.7);
        if (0 == finetag) {
//            Log.e(">>>", "当前片段不存在手势");
            System.out.println("当前片段不存在手势");
        } else {
//            Log.e(">>>", "手势点：" + ma.pointstartindex + " " + ma.pointendindex);
            System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
            orippg.x = nortools.innerscale(orippg.x);
            orippg.y = nortools.innerscale(orippg.y);
            orippg = ma.setppgsegment(orippg);
            Motion motion = ma.setmotionsegment(motions);
            
            samplefeature = return_feature(orippg, motion);
            motion = null;

        }
        orippg = null;
        butterppg = null;
        icappg = null;
		ma = null;
		nortools = null;
        iatools = null;
        System.gc();
        return samplefeature;
    }


}
