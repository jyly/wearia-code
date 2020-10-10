package IA;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

public class IA_main {

	
	static filecontrol files = new filecontrol();
	static IAtool iatools = new IAtool();
	static Normal_tool nortools = new Normal_tool();
	static MAfind ma = new MAfind();
	static Featurecontrol featurecontrol = new Featurecontrol();

	public static void main(String[] args) {


		String dirpath = "./selected_oridata/";
		all_feature(dirpath);
//		all_madata(dirpath);
	}

	public static void all_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		// 文件夹中提取的文件数
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			File[] samplefileset = objdir.listFiles();
			ArrayList<double[]> featureset = new ArrayList<double[]>();
			for (File sample : samplefileset) {
				double[] sampleFeature = single_feature(sample);
				if (sampleFeature != null) {
					featureset.add(sampleFeature);
				}
			}
			if (featureset.size() > 0) {
				System.out.println("当前第" + objnum + "个类别手势片段数：" + featureset.size());
				filenum.add(featureset.size());
				String featurefile = "./selected_feature/" + objdir.getName() + ".csv";
				objnum++;
				files.featurewrite(featureset, featurefile);
			}
		}
		System.out.println("filenum:");
		for (int i = 0; i < filenum.size(); i++) {
			System.out.print(filenum.get(i) + ",");
		}
	}

	public static double[] single_feature(File filepath) {

		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);

		double[] samplefeature = null;
		ppgs.x = nortools.meanfilt(ppgs.x, 20);
		ppgs.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
		butterppg.x = nortools.butterworth_highpass(ppgs.x, 200, 2);
		butterppg.y = nortools.butterworth_highpass(ppgs.y, 200, 2);

//		butterppg.x = nortools.butterworth_bandpass(ppgs.x, 200, 3,1);
//		butterppg.y = nortools.butterworth_bandpass(ppgs.y, 200, 3,1);

		// 做快速主成分分析
		Ppg icappg = iatools.fastica(butterppg);

		// 根据峰值判断那条手势信号和脉冲信号
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(icappg.x, icappg.y, featurefile);

		MAfind ma = new MAfind();
		// 细粒度手势分析，判断手势区间
//		int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5);
		if (0 == finetag) {
//			Log.e(">>>", "当前片段不存在手势");
			System.out.println("当前片段不存在手势");
		} else {
//			Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			Normal_tool normal = new Normal_tool();

			if (ppgs.x[0] > 100) {

				ppgs.x = normal.innerscale(ppgs.x);
				ppgs.y = normal.innerscale(ppgs.y);

			}
//			ppgs.x = normal.outterscale(ppgs.x);
//			ppgs.y = normal.outterscale(ppgs.y);
			normal = null;
			ppgs = ma.setppgsegment(ppgs);
			butterppg = ma.setppgsegment(butterppg);
			icappg = ma.setppgsegment(icappg);
			Motion motion = files.orimotionread(filepath);
			motion = ma.setmotionsegment(motion);
			ma = null;

			Featurecontrol featurecontrol = new Featurecontrol();
			samplefeature = featurecontrol.return_feature(butterppg, motion);
//			samplefeature = featurecontrol.return_feature(ppgs, motion);
		}

		return samplefeature;
	}

	public static void all_madata(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		// 文件夹中提取的文件数
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			File[] samplefileset = objdir.listFiles();
			ArrayList<double[][]> madata = new ArrayList<double[][]>();
			for (File sample : samplefileset) {
				double[][] sampledata = single_data(sample);
				if (sampledata != null) {
					System.out.println(sampledata[0][0]);
					madata.add(sampledata);
				}
			}
			if (madata.size() > 0) {
				System.out.println("当前第" + objnum + "个类别手势片段数：" + madata.size());
				filenum.add(madata.size());
				String featurefile = "./selected_madata/" + objdir.getName() + ".csv";
				objnum++;
				files.madatawrite(madata, featurefile);
			}
		}
		System.out.println("filenum:");
		for (int i = 0; i < filenum.size(); i++) {
			System.out.print(filenum.get(i) + ",");
		}
	}

	public static double[][] single_data(File filepath) {

		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);

		double[][] sampledata = null;
		ppgs.x = nortools.meanfilt(ppgs.x, 20);
		ppgs.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
		butterppg.x = nortools.butterworth_highpass(ppgs.x, 200, 2);
		butterppg.y = nortools.butterworth_highpass(ppgs.y, 200, 2);

		// 做快速主成分分析
		Ppg icappg = iatools.fastica(butterppg);

		// 根据峰值判断那条手势信号和脉冲信号
		icappg = iatools.machoice(icappg);

		MAfind ma = new MAfind();
		// 细粒度手势分析，判断手势区间

		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5);
		if (0 == finetag) {
//			Log.e(">>>", "当前片段不存在手势");
			System.out.println("当前片段不存在手势");
		} else {
//			Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			Normal_tool normal = new Normal_tool();

			if (ppgs.x[0] > 100) {

				ppgs.x = normal.innerscale(ppgs.x);
				ppgs.y = normal.innerscale(ppgs.y);

			}
//			System.out.println("ppgs.x：" + ppgs.x[0]);
			sampledata = new double[8][];
			ppgs = ma.setppgsegment(ppgs);

			Motion motion = files.orimotionread(filepath);
			motion = ma.setmotionsegment(motion);

			motion.accx = normal.increto_2(motion.accx);
			motion.accy = normal.increto_2(motion.accy);
			motion.accz = normal.increto_2(motion.accz);
			motion.gyrx = normal.increto_2(motion.gyrx);
			motion.gyry = normal.increto_2(motion.gyry);
			motion.gyrz = normal.increto_2(motion.gyrz);
			normal = null;
			ma = null;

			sampledata[0] = ppgs.x;
			sampledata[1] = ppgs.y;
			sampledata[2] = motion.accx;
			sampledata[3] = motion.accy;
			sampledata[4] = motion.accz;
			sampledata[5] = motion.gyrx;
			sampledata[6] = motion.gyry;
			sampledata[7] = motion.gyrz;
		}

		return sampledata;
	}

}
