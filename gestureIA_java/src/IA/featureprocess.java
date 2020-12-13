package IA;

import java.io.File;
import java.util.ArrayList;

public class featureprocess {
	static filecontrol files = new filecontrol();
	IAtool iatools = new IAtool();
	Normal_tool nortools = new Normal_tool();
	MAfind ma = new MAfind();
	Featurecontrol featurecontrol = new Featurecontrol();

	// ����Ծ�ֹ״̬�µ���������40��
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
//					double[] sampleFeature = single_feature(sample);
					double[] sampleFeature = null;
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
	
	// �����Ƶ�������360��
	public void all_feature(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		// �ļ�������ȡ���ļ���
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			// ����Ŀ���ļ����������ļ�
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
		
		if (ppgs.x.length < 800 || ppgs.y.length < 800) {
			return samplefeature;
		}

		Ppg orippg = new Ppg();

		orippg.x = nortools.minmaxscale(ppgs.x);
		orippg.y = nortools.minmaxscale(ppgs.y);
//		orippg.x = nortools.meanfilt(ppgs.x, 20);
//		orippg.y = nortools.meanfilt(ppgs.y, 20);

		Ppg butterppg = new Ppg();
//		//��ԭʼ��ppg�ͺ���butterworth��ȡ
//		butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
//		butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);

		butterppg.x = nortools.butterworth_bandpass(orippg.x, 200, 2, 5);
		butterppg.y = nortools.butterworth_bandpass(orippg.y, 200, 2, 5);

		butterppg.x = nortools.minmaxscale(butterppg.x);
		butterppg.y = nortools.minmaxscale(butterppg.y);

		// ���������ɷַ���
		Ppg icappg = iatools.fastica(butterppg);
//		 ���ݷ�ֵ�ж����������źź������ź�
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(icappg.x, icappg.y, featurefile);

		MAfind ma = new MAfind();
		Normal_tool normal = new Normal_tool();
		Featurecontrol featurecontrol = new Featurecontrol();
		// ϸ�������Ʒ������ж���������
		int finetag = ma.fine_grained_segment(icappg.x, 200, 0.7);//��������ȡ����
//		int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5, 0.7);//��������ȡ����
//		int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.6);//�����������ƶ���ȡ����
		if (0 == finetag) {
//				Log.e(">>>", "��ǰƬ�β���������");
			System.out.println("��ǰƬ�β���������");
		} else {
//				Log.e(">>>","���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);

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
//			System.out.println("��������" + samplefeature.length);
		}
		return samplefeature;
	}

}
