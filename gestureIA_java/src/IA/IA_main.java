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
	static normal_tool nortools = new normal_tool();
	static MAfind ma = new MAfind();

	public static void main(String[] args) {

		String dirpath = "./selected_oridata/";
		all_feature(dirpath);

//		files.featurewrite(featureset, featurefile);
//		String filepath = "./testdata/2020-06-16-08-11-54.csv";
//		File File = new File(filepath);
//		single_feature(File);

//		String modelpath = "./test.model";
//		String informscoreString = "[32,34,33,39,35,36,38,37,50,10,68,73,11,51,67,1,2,52,59,20]";
//		String ldamatrixString = "[[6.04079303e+03],[-4.41543626e+03],[-6.62634335e+02],[-8.38821195e+02],[1.97501034e+03],[2.25793830e+03],[ 2.74099152e+03],[-3.65587825e+03],[-2.35520457e+04],[ 5.87755757e+05],[ 1.75773472e+03],[ 6.98597705e+01],[-5.93221556e+05],[ 2.07077389e+04],[-2.63030546e+01],[ 8.30488278e+04],[-8.28362445e+04],[-1.44930568e-01],[ 1.72691035e+02],[-1.39858221e+02]]";
//
//		// 转为数字矩阵
//		String[] tempinformscore = new String[20];
//		String[] templdamatrixString = new String[20 * 1];
//
//		int[] informscore = new int[20];
//		double[][] ldamatrix = new double[20][1];
//
//		tempinformscore = informscoreString.replace("[", "").replace("]", "").split(",");
//		templdamatrixString = ldamatrixString.replace("[", "").replace("]", "").split(",");
//		System.out.println("特征选择：" + tempinformscore[0]);
//		for (int i = 0; i < 20; i++) {
//			informscore[i] = Integer.valueOf(tempinformscore[i]);
//		}
//
//		for (int i = 0; i < 20; i++) {
//			for (int j = 0; j < 1; j++) {
//				ldamatrix[i][j] = Double.valueOf(templdamatrixString[i * 1 + j]);
//			}
//		}
//
//		double[] selsectfeature = new double[20];
//		for (int i = 0; i < 20; i++) {
//			selsectfeature[i] = features.features.get(informscore[i]);
//		}
//		svm_node[] finalfeature = new svm_node[1];
//		for (int i = 0; i < 1; i++) {
//			double temp = 0;
//			for (int j = 0; j < 20; j++) {
//				temp += selsectfeature[j] * ldamatrix[j][i];
//			}
//			finalfeature[i] = new svm_node();
//			finalfeature[i].index=i;
//
//			finalfeature[i].value = temp;
//		}
//
//		svm_predict p = new svm_predict();
//		svm svms = new svm();
//		svm_model models = new svm_model();
//		try {
//			models = svms.svm_load_model(modelpath);
//			double result = svms.svm_predict(models, finalfeature);
//			System.out.println("result：" + result);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
	}

	public static void all_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			File[] samplefileset = objdir.listFiles();

			ArrayList<feature> featureset = new ArrayList<feature>();
			for (File sample : samplefileset) {
				feature sampleFeature = single_feature(sample);
				if (0 != sampleFeature.features.size()) {
					featureset.add(sampleFeature);
				}
			}
			System.out.println("当前第" + objnum + "个类别手势片段数：" + featureset.size());
			filenum.add(featureset.size());
			String featurefile = "./selected_feature/" + String.valueOf(objnum) + ".csv";
			objnum++;
			files.featurewrite(featureset, featurefile);

//			featureset = new ArrayList<feature>();
//			for (File sample : samplefileset) {
//				feature sampleFeature = single_feature_2(sample);
//				if (0 != sampleFeature.features.size()) {
//					featureset.add(sampleFeature);
//				}
//			}
//			System.out.println("当前第" + objnum + "个类别手势片段数：" + featureset.size());
//			filenum.add(featureset.size());
//			featurefile = "./selected_feature/" + String.valueOf(objnum) + ".csv";
//			objnum++;
//			files.featurewrite(featureset, featurefile);

		}
		System.out.println("filenum:");
		for (int i = 0; i < filenum.size(); i++) {
			System.out.print(filenum.get(i) + ",");
		}
	}

	public static feature single_feature(File filepath) {
		feature singleFeature = new feature();
		System.out.println(filepath);
		ppg ppgs = files.orippgread(filepath);

		
		System.out.println("reading success");

		double[] orippgx = nortools.meanfilt(nortools.arraytomatrix(ppgs.x), 20);
		double[] orippgy = nortools.meanfilt(nortools.arraytomatrix(ppgs.y), 20);

		int coarsetag = ma.coarse_grained_detect(orippgx);
		System.out.println("coarsetag:" + coarsetag);
//		if(1==tag) {}

//		//对原始的ppg型号做butterworth提取
		double[] butterppgx = nortools.butterworth_highpass(orippgx, 200, 2);
		double[] butterppgy = nortools.butterworth_highpass(orippgy, 200, 2);
//
		ppgs.x = nortools.matrixtoarray(nortools.array_dataselect(orippgx, 300, orippgx.length - 300));
		ppgs.y = nortools.matrixtoarray(nortools.array_dataselect(orippgy, 300, orippgy.length - 300));

		ppg butterppg = new ppg();
		butterppg.x = nortools.matrixtoarray(nortools.array_dataselect(butterppgx, 300, butterppgx.length - 300));
		butterppg.y = nortools.matrixtoarray(nortools.array_dataselect(butterppgy, 300, butterppgx.length - 300));
		// 做快速主成分分析
		butterppg = iatools.fastica(butterppg);
//
//		// 根据峰值判断那条手势信号和脉冲信号
		butterppg = iatools.machoice(butterppg);
//
		int finetag = ma.fine_grained_segment(nortools.arraytomatrix(butterppg.x), 200, 1);

		if (0 == finetag) {
			System.out.println("当前片段不存在手势");
		} else {
			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
			ppgs = ma.setMAsegment(ppgs);
			Motion motion = files.orimotionread(filepath);
			int datalen=motion.accx.size();
			
			motion.accx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accx),150, datalen - 150));
			motion.accy = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accy),150, datalen - 150));
			motion.accz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.accz),150, datalen - 150));
			datalen=motion.gyrx.size();
			System.out.println("gyrx len：" + datalen);
			motion.gyrx = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrx),150, datalen - 150));
			motion.gyry = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyry),150, datalen - 150));
			motion.gyrz = nortools.matrixtoarray(nortools.array_dataselect(nortools.arraytomatrix(motion.gyrz),150, datalen - 150));

			motion=ma.setmotionsegment(motion);
			singleFeature.ppg_feature(nortools.arraytomatrix(ppgs.x));
			singleFeature.ppg_feature(nortools.arraytomatrix(ppgs.y));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.accx));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.accy));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.accz));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.gyrx));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.gyry));
			singleFeature.motion_feature(nortools.arraytomatrix(motion.gyrz));
		}

		return singleFeature;
	}

//	public static feature single_feature_2(File filepath) {
//		feature singleFeature = new feature();
//		System.out.println(filepath);
//		ppg ppgs = files.orippgread(filepath);
//		Motion motion = files.orimotionread(filepath);
//		System.out.println("reading success");
//
//		double[] orippgx = nortools.meanfilt(nortools.arraytomatrix(ppgs.x), 20);
//		double[] orippgy = nortools.meanfilt(nortools.arraytomatrix(ppgs.y), 20);
//
//		int coarsetag = ma.coarse_grained_detect(orippgx);
//		System.out.println("coarsetag:" + coarsetag);
////		if(1==tag) {}
//
////		//对原始的ppg型号做butterworth提取
//		double[] butterppgx = nortools.butterworth_highpass(orippgx, 200, 2);
//		double[] butterppgy = nortools.butterworth_highpass(orippgy, 200, 2);
//
//		ppg butterppg = new ppg();
//		butterppg.x = nortools.matrixtoarray(nortools.array_dataselect(butterppgx, 300, butterppgx.length - 300));
//		butterppg.y = nortools.matrixtoarray(nortools.array_dataselect(butterppgy, 300, butterppgx.length - 300));
//		// 做快速主成分分析
//		butterppg = iatools.fastica(butterppg);
////
////		// 根据峰值判断那条手势信号和脉冲信号
//		butterppg = iatools.machoice(butterppg);
////
//		int finetag = ma.fine_grained_segment(nortools.arraytomatrix(butterppg.x), 200, 1);
//
//		if (0 == finetag) {
//			System.out.println("当前片段不存在手势");
//		} else {
//			System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);
//			ppgs = ma.setslience(ppgs);
//
//			singleFeature.ppg_feature(nortools.arraytomatrix(ppgs.x));
//			singleFeature.ppg_feature(nortools.arraytomatrix(ppgs.y));
////			if (ppgs.x.size() > 300) {
//			//			}
//		}
//
//		return singleFeature;
//	}
}
