package IA;

import java.io.File;
import java.util.ArrayList;

public class featureprocess {
	static filecontrol files = new filecontrol();
	IAtool iatools = new IAtool();
	Normal_tool nortools = new Normal_tool();
	MAfind ma = new MAfind();
	Featurecontrol featurecontrol = new Featurecontrol();

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
					double[] sampleFeature = single_feature(sample);
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

	public void all_feature(String dirpath) {
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

	public double[] single_feature(File filepath) {
		double[] samplefeature = null;
		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);
		if(ppgs.x.length<800||ppgs.y.length<800) {
			return samplefeature;
		}
		
		Ppg orippg = new Ppg();
		orippg.x = nortools.meanfilt(ppgs.x, 20);
		orippg.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
		butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
		butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);

//		butterppg.x = nortools.butterworth_bandpass(ppgs.x, 200, 2,10);
//		butterppg.y = nortools.butterworth_bandpass(ppgs.y, 200, 2,10);
		
		// 做快速主成分分析
		Ppg icappg = iatools.fastica(butterppg);
//		 根据峰值判断那条手势信号和脉冲信号
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(butterppg.x, butterppg.y, featurefile);

		MAfind ma = new MAfind();
		Normal_tool normal = new Normal_tool();
		Featurecontrol featurecontrol = new Featurecontrol();
		// 细粒度手势分析，判断手势区间
		int finetag = ma.fine_grained_segment(icappg.x, 200, 0.7);
//		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1);
//		int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.6);
		if (0 == finetag) {
//				Log.e(">>>", "当前片段不存在手势");
			System.out.println("当前片段不存在手势");
		} else {
//				Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);

			orippg.x = normal.innerscale(orippg.x);
			orippg.y = normal.innerscale(orippg.y);
//			orippg.x=normal.standardscale(orippg.x);
//			orippg.x=normal.standardscale(orippg.y);
//			ppgs.x = normal.innerscale(ppgs.x);
//			ppgs.y = normal.innerscale(ppgs.y);
			
			orippg = ma.setppgsegment(orippg);
//			butterppg = ma.setppgsegment(butterppg);
//			ppgs = ma.setppgsegment(ppgs);
//			icappg = ma.setppgsegment(icappg);

//			Motion motion = files.orimotionread(filepath);
//			motion = ma.setmotionsegment(motion);
//			samplefeature = featurecontrol.return_feature(butterppg, motion);
			samplefeature = featurecontrol.return_feature(orippg);
//			System.out.println("特征数：" + samplefeature.length);
		}

		return samplefeature;
	}

}
