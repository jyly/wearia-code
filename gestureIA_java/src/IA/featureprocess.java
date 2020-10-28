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
		// �ļ�������ȡ�����
		for (File objdir : objdirName) {
			File[] fileset = objdir.listFiles();
			ArrayList<double[]> featureset = new ArrayList<double[]>();
			// �ļ�������ȡ�����,��ʷ������
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
				System.out.println("��ǰ��" + objnum + "�����ֹƬ������" + featureset.size());
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
		// �ļ�������ȡ���ļ���
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
				System.out.println("��ǰ��" + objnum + "���������Ƭ������" + featureset.size());
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
//		//��ԭʼ��ppg�ͺ���butterworth��ȡ
		butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
		butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);

//		butterppg.x = nortools.butterworth_bandpass(ppgs.x, 200, 2,10);
//		butterppg.y = nortools.butterworth_bandpass(ppgs.y, 200, 2,10);
		
		// ���������ɷַ���
		Ppg icappg = iatools.fastica(butterppg);
//		 ���ݷ�ֵ�ж����������źź������ź�
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(butterppg.x, butterppg.y, featurefile);

		MAfind ma = new MAfind();
		Normal_tool normal = new Normal_tool();
		Featurecontrol featurecontrol = new Featurecontrol();
		// ϸ�������Ʒ������ж���������
		int finetag = ma.fine_grained_segment(icappg.x, 200, 0.7);
//		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1);
//		int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.6);
		if (0 == finetag) {
//				Log.e(">>>", "��ǰƬ�β���������");
			System.out.println("��ǰƬ�β���������");
		} else {
//				Log.e(">>>","���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);

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
//			System.out.println("��������" + samplefeature.length);
		}

		return samplefeature;
	}

}
