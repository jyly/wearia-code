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
	static Featurecontrol featurecontrol=new Featurecontrol();
	public static void main(String[] args) {
		String dirpath = "./selected_oridata/";
		all_feature(dirpath);
	}

	public static void all_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		//文件夹中提取的文件数
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			File[] samplefileset = objdir.listFiles();
			ArrayList<double[]> featureset = new ArrayList<double[]>();
			for (File sample : samplefileset) {
				double[] sampleFeature = single_feature(sample);
				if ( sampleFeature!=null) {
					featureset.add(sampleFeature);
				}
			}
			System.out.println("当前第" + objnum + "个类别手势片段数：" + featureset.size());
			filenum.add(featureset.size());
			String featurefile = "./selected_feature/" +objdir.getName() + ".csv";
			objnum++;
			files.featurewrite(featureset, featurefile);
		}
		System.out.println("filenum:");
		for (int i = 0; i < filenum.size(); i++) {
			System.out.print(filenum.get(i) + ",");
		}
	}

	public static double[] single_feature(File filepath) {
		
		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);

		
		MAfind ma = new MAfind();

		double[] samplefeature= null;
		ppgs.x = nortools.meanfilt(ppgs.x, 20);
		ppgs.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//对原始的ppg型号做butterworth提取
		butterppg.x = nortools.butterworth_highpass(ppgs.x, 200, 2);
		butterppg.y = nortools.butterworth_highpass(ppgs.y, 200, 2);

		int inter=600;
		butterppg.x = nortools.array_dataselect(butterppg.x, inter, butterppg.x.length - inter);
		butterppg.y = nortools.array_dataselect(butterppg.y, inter, butterppg.y.length - inter);
		// 做快速主成分分析
		Ppg icappg = iatools.fastica(butterppg);
		// 根据峰值判断那条手势信号和脉冲信号
		icappg = iatools.machoice(icappg);
		//细粒度手势分析，判断手势区间
		int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
		if (0 == finetag) {
//			Log.e(">>>", "当前片段不存在手势");
			System.out.println("当前片段不存在手势");
		} else {
//			Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			ppgs.x = nortools.array_dataselect(ppgs.x, inter, ppgs.x.length - inter);
			ppgs.y = nortools.array_dataselect(ppgs.y, inter, ppgs.y.length - inter);

			ppgs = ma.setppgsegment(ppgs);
			butterppg = ma.setppgsegment(butterppg);
			icappg = ma.setppgsegment(icappg);
			Motion motion = files.orimotionread(filepath);
			int datalen=motion.accx.length;
			motion.accx = nortools.array_dataselect(motion.accx,inter/2,  datalen- inter/2);
			motion.accy = nortools.array_dataselect(motion.accy,inter/2, datalen - inter/2);
			motion.accz = nortools.array_dataselect(motion.accz,inter/2, datalen - inter/2);
			motion.gyrx = nortools.array_dataselect(motion.gyrx,inter/2, datalen - inter/2);
			motion.gyry = nortools.array_dataselect(motion.gyry,inter/2, datalen - inter/2);
			motion.gyrz = nortools.array_dataselect(motion.gyrz,inter/2, datalen - inter/2);
			motion=ma.setmotionsegment(motion);
			ma=null;
//			samplefeature=return_feature(ppgs,motion);
            Featurecontrol featurecontrol = new Featurecontrol();

			samplefeature=featurecontrol.return_feature(ppgs,motion,butterppg,icappg);
		}

		return samplefeature;
	}

}
