package IA;

import java.io.File;
import java.util.ArrayList;

public class combine {
	filecontrol files = new filecontrol();
	IAtool iatools = new IAtool();
	Normal_tool nortools = new Normal_tool();
	MAfind ma = new MAfind();
	Featurecontrol featurecontrol = new Featurecontrol();
	// 求手势的特征，360组
		public void all_madata(String dirpath) {
			File dirFile = new File(dirpath);
			File[] objdirName = dirFile.listFiles();
			// 文件夹中提取的文件数
			ArrayList<Integer> filenum = new ArrayList<Integer>();
			int objnum = 1;
			for (File objdir : objdirName) {
				File[] samplefileset = objdir.listFiles();
				ArrayList<double[][]> madata = new ArrayList<double[][]>();
				ArrayList<double[]> featureset = new ArrayList<double[]>();
				for (File sample : samplefileset) {
					combines_class single = single_data(sample);
					if (single.sampledata != null) {
						madata.add(single.sampledata);
						featureset.add(single.samplefeature);
					}
				}
				if (madata.size() > 0) {
					System.out.println("当前第" + objnum + "个类别手势片段数：" + madata.size());
					filenum.add(madata.size());
					String featurefile = "./selected_madata/" + objdir.getName() + ".csv";
					objnum++;
					files.madatawrite(madata, featurefile);
					featurefile = "./selected_feature/" + objdir.getName() + ".csv";
					files.featurewrite(featureset, featurefile);
				}
			}
			System.out.println("filenum:");
			for (int i = 0; i < filenum.size(); i++) {
				System.out.print(filenum.get(i) + ",");
			}
		}
		public combines_class single_data(File filepath) {
			combines_class single=new combines_class();
			
			single.sampledata = null;
			single.samplefeature = null;
			
			System.out.println(filepath);
			Ppg ppgs = files.orippgread(filepath);
			
			if(ppgs.x.length<800||ppgs.y.length<800) {
				return single;
			}
				
			Ppg orippg = new Ppg();
			orippg.x=nortools.minmaxscale(ppgs.x);
			orippg.y=nortools.minmaxscale(ppgs.y);
			
			
			Ppg butterppg = new Ppg();
//			//对原始的ppg型号做butterworth提取
			butterppg.x = nortools.butterworth_bandpass(orippg.x, 200, 2,5);
			butterppg.y = nortools.butterworth_bandpass(orippg.y, 200, 2,5);
			
			butterppg.x=nortools.minmaxscale(butterppg.x);
			butterppg.y=nortools.minmaxscale(butterppg.y);
			
			// 做快速主成分分析
			Ppg icappg = iatools.fastica(butterppg);

			// 根据峰值判断那条手势信号和脉冲信号
			icappg = iatools.machoice(icappg);

			
			MAfind ma = new MAfind();
			// 细粒度手势分析，判断手势区间
			
			int finetag = ma.fine_grained_segment(icappg.x, 200, 1);
//			 int finetag = ma.fine_grained_segment_2(icappg.x, 200, 1.5,0.7);
//			int finetag = ma.fine_grained_segment_3(icappg.x, 200, 0.3);
			if (0 == finetag) {
//				Log.e(">>>", "当前片段不存在手势");
				System.out.println("当前片段不存在手势");
			} else {
//				Log.e(">>>","手势点：" + ma.pointstartindex + " " + ma.pointendindex);
				System.out.println("手势点：" + ma.pointstartindex + " " + ma.pointendindex);

				single.sampledata = new double[8][];
				ppgs.x = nortools.meanfilt(ppgs.x, 20);
				ppgs.y = nortools.meanfilt(ppgs.y, 20);
				ppgs = ma.setsegment(ppgs);
				
//				ppgs.x=nortools.smallscale(ppgs.x);
//				ppgs.y=nortools.smallscale(ppgs.y);
			
				ppgs.x=nortools.minmaxscale(ppgs.x);
				ppgs.y=nortools.minmaxscale(ppgs.y);
				
				
				Motion motion = files.orimotionread(filepath);
				motion.accx = nortools.meanfilt(motion.accx, 20);
				motion.accy = nortools.meanfilt(motion.accy, 20);
				motion.accz = nortools.meanfilt(motion.accz, 20);
				motion.gyrx = nortools.meanfilt(motion.gyrx, 20);
				motion.gyry = nortools.meanfilt(motion.gyry, 20);
				motion.gyrz = nortools.meanfilt(motion.gyrz, 20);
				motion = ma.setsegment(motion);
				ma = null;

				
				motion.accx=nortools.minmaxscale(motion.accx);
				motion.accy=nortools.minmaxscale(motion.accy);
				motion.accz=nortools.minmaxscale(motion.accz);
				motion.gyrx=nortools.minmaxscale(motion.gyrx);
				motion.gyry=nortools.minmaxscale(motion.gyry);
				motion.gyrz=nortools.minmaxscale(motion.gyrz);		
				
				single.sampledata[0] = ppgs.x;
				single.sampledata[1] = ppgs.y;
				single.sampledata[2] = motion.accx;
				single.sampledata[3] = motion.accy;
				single.sampledata[4] = motion.accz;
				single.sampledata[5] = motion.gyrx;
				single.sampledata[6] = motion.gyry;
				single.sampledata[7] = motion.gyrz;
				

				single.samplefeature = featurecontrol.return_feature(ppgs, motion);
			}

			return single;
		}
}
