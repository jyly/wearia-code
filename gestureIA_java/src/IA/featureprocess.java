package IA;

import java.io.File;
import java.util.ArrayList;

public class featureprocess {
	static filecontrol files = new filecontrol();
	IAtool iatools = new IAtool();
	Normal_tool nortools = new Normal_tool();
	MAfind ma = new MAfind();
	Featurecontrol featurecontrol = new Featurecontrol();

	// 求相对静止状态下的特征，用40组
	public void static_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		// 文件夹中提取的类别
		for (File objdir : objdirName) {
			File[] fileset = objdir.listFiles();
			ArrayList<double[]> featureset = new ArrayList<double[]>();
			// 文件夹中提取的类别,历史遗留类
			for (File samplefileset : fileset) {
				File[] samples = samplefileset.listFiles();
				for (File sample : samples) {
//					double[] sampleFeature = single_feature(sample);
					double[] sampleFeature = null;
					if (sampleFeature != null) {
						featureset.add(sampleFeature);
					}
				}
			}
			if (featureset.size() > 0) {
				System.out.println("当前第" + objnum + "个类别静止片段数：" + featureset.size());
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
	
	// 求手势的特征，360组
	public void all_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		// 文件夹中提取的文件数
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			// 遍历目标文件夹下所有文件
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

	public double[] single_feature(File filepath) {
		double[] samplefeature = null;
		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);
		
		if (ppgs.x.length < 800 || ppgs.y.length < 800) {
			return samplefeature;
		}

		Ppg orippg = new Ppg();

		orippg.x = nortools.minmaxscale(ppgs.x);
		orippg.y = nortools.minmaxscale(ppgs.y);
//		orippg.x = nortools.meanfilt(ppgs.x, 20);
//		orippg.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
//		butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
//		butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);

		butterppg.x = nortools.butterworth_bandpass(orippg.x, 200, 2, 5);
		butterppg.y = nortools.butterworth_bandpass(orippg.y, 200, 2, 5);

		butterppg.x = nortools.minmaxscale(butterppg.x);
		butterppg.y = nortools.minmaxscale(butterppg.y);

		// 做快速主成分分析
		Ppg icappg = iatools.fastica(butterppg);
//		 根据峰值判断那条手势信号和脉冲信号
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(icappg.x, icappg.y, featurefile);

		MAfind ma = new MAfind();
		Normal_tool normal = new Normal_tool();
		Featurecontrol featurecontrol = new Featurecontrol();
		// 细粒度手势分析，判断手势区间
		int finetag = ma.fine_grained_segment(icappg.x, 200, 0.7);//旧手势提取方法
//		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5, 0.7);//新手势提取方法
//		int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.6);//粗粒度无手势段提取方法
		if (0 == finetag) {
//				Log.e(">>>", "当前片段不存在手势");
			System.out.println("当前片段不存在手势");
		} else {
//				Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);

			ppgs.x = nortools.meanfilt(ppgs.x, 20);
			ppgs.y = nortools.meanfilt(ppgs.y, 20);
			ppgs = ma.setsegment(ppgs);
			
//			ppgs.x = nortools.changehz(ppgs.x, 10);
//			ppgs.y = nortools.changehz(ppgs.y, 10);
			
//			System.out.printf("newlens"+ppgs.x.length);

			Motion motion = files.orimotionread(filepath);
			motion.accx = nortools.meanfilt(motion.accx, 20);
			motion.accy = nortools.meanfilt(motion.accy, 20);
			motion.accz = nortools.meanfilt(motion.accz, 20);
			motion.gyrx = nortools.meanfilt(motion.gyrx, 20);
			motion.gyry = nortools.meanfilt(motion.gyry, 20);
			motion.gyrz = nortools.meanfilt(motion.gyrz, 20);
			motion = ma.setsegment(motion);
			
			samplefeature = featurecontrol.return_feature(ppgs, motion);
//			samplefeature = featurecontrol.return_feature(ppgs);
//			System.out.println("特征数：" + samplefeature.length);
		}
		return samplefeature;
	}

}
