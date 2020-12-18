package IA;

import java.io.File;
import java.util.ArrayList;

public class dataprocess {
	filecontrol files = new filecontrol();
	IAtool iatools = new IAtool();
	Normal_tool nortools = new Normal_tool();
	MAfind ma = new MAfind();
	Featurecontrol featurecontrol = new Featurecontrol();

	// ����Ծ�ֹ״̬�µ����ݣ���40��
	public void static_data(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		// �ļ�������ȡ������û���
		for (File objdir : objdirName) {
			File[] fileset = objdir.listFiles();
			int madatasize = 0;
			// �ļ�������ȡ�����,��ʷ�����࣬������
			for (File samplefileset : fileset) {
				File[] samples = samplefileset.listFiles();
				for (File sample : samples) {
					double[][] sampledata = single_data(sample);
					if (sampledata != null) {
						madatasize=madatasize+1;
						String featurefile = "./selected_madata/" + objdir.getName()+"-"+sample.getName();
						objnum++;
						files.madatawrite(sampledata, featurefile);
						
					}
				}
			}
			if (madatasize> 0) {
				System.out.println("��ǰ��" + objnum + "�����ֹƬ������" + madatasize);
				filenum.add(madatasize);
				objnum++;
			}
		}
		System.out.println("filenum:");
		for (int i = 0; i < filenum.size(); i++) {
			System.out.print(filenum.get(i) + ",");
		}
	}

	

	// �����Ƶ�������360��
	public void all_madata(String dirpath) {
		File dirFile = new File(dirpath);
		File[] objdirName = dirFile.listFiles();
		// �ļ�������ȡ���ļ���
		ArrayList<Integer> filenum = new ArrayList<Integer>();
		int objnum = 1;
		for (File objdir : objdirName) {
			File[] samplefileset = objdir.listFiles();
			ArrayList<double[][]> madata = new ArrayList<double[][]>();
			for (File sample : samplefileset) {
				double[][] sampledata = single_data(sample);
				if (sampledata != null) {
					madata.add(sampledata);
				}
			}
			if (madata.size() > 0) {
				System.out.println("��ǰ��" + objnum + "���������Ƭ������" + madata.size());
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

	public double[][] single_data(File filepath) {
		double[][] sampledata = null;
		System.out.println(filepath);
		Ppg ppgs = files.orippgread(filepath);
		
		if(ppgs.x.length<800||ppgs.y.length<800) {
			return sampledata;
		}
			
		Ppg orippg = new Ppg();
//		orippg.x = nortools.meanfilt(ppgs.x, 20);
//		orippg.y = nortools.meanfilt(ppgs.y, 20);
		orippg.x=nortools.minmaxscale(ppgs.x);
		orippg.y=nortools.minmaxscale(ppgs.y);
		
		
		Ppg butterppg = new Ppg();
//		//��ԭʼ��ppg�ͺ���butterworth��ȡ
//		butterppg.x = nortools.butterworth_bandpass(ppgs.x, 200, 2,10);
//		butterppg.y = nortools.butterworth_bandpass(ppgs.y, 200, 2,10);
		butterppg.x = nortools.butterworth_bandpass(orippg.x, 200, 2,5);
		butterppg.y = nortools.butterworth_bandpass(orippg.y, 200, 2,5);
		
		butterppg.x=nortools.minmaxscale(butterppg.x);
		butterppg.y=nortools.minmaxscale(butterppg.y);
		
		// ���������ɷַ���
		Ppg icappg = iatools.fastica(butterppg);

		// ���ݷ�ֵ�ж����������źź������ź�
		icappg = iatools.machoice(icappg);

//		String featurefile = "./butter/" + filepath.getName();
//		files.datawrite(icappg.x, icappg.y, featurefile);
		
		MAfind ma = new MAfind();
		// ϸ�������Ʒ������ж���������
		
//		int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
		 int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5,0.7);
//		int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.3);
		if (0 == finetag) {
//			Log.e(">>>", "��ǰƬ�β���������");
			System.out.println("��ǰƬ�β���������");
		} else {
//			Log.e(">>>","���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);
			System.out.println("���Ƶ㣺" + ma.pointstartindex + " " + ma.pointendindex);


//			orippg.x = normal.innerscale(orippg.x);
//			orippg.y = normal.innerscale(orippg.y);

						
//			ppgs.x = normal.innerscale(ppgs.x);
//			ppgs.y = normal.innerscale(ppgs.y);
//			butterppg.x = nortools.butterworth_highpass(orippg.x, 200, 2);
//			butterppg.y = nortools.butterworth_highpass(orippg.y, 200, 2);

			
//			String featurefile = "./butter/" + filepath.getName();
//			files.datawrite(butterppg.x, butterppg.y, featurefile);


			sampledata = new double[8][];
			ppgs.x = nortools.meanfilt(ppgs.x, 20);
			ppgs.y = nortools.meanfilt(ppgs.y, 20);
			ppgs = ma.setsegment(ppgs);

			Motion motion = files.orimotionread(filepath);
			motion.accx = nortools.meanfilt(motion.accx, 20);
			motion.accy = nortools.meanfilt(motion.accy, 20);
			motion.accz = nortools.meanfilt(motion.accz, 20);
			motion.gyrx = nortools.meanfilt(motion.gyrx, 20);
			motion.gyry = nortools.meanfilt(motion.gyry, 20);
			motion.gyrz = nortools.meanfilt(motion.gyrz, 20);
			motion = ma.setsegment(motion);
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
