package IA;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import java.text.DecimalFormat;

public class MAfind {
	public int pointstartindex = 0;
	public int pointendindex = 0;

	// pggpass方案中计算能量的方案
	public ArrayList<Double> energycal(ArrayList<Double> data, int win, double threshold) {
		ArrayList<Double> energy = new ArrayList<Double>();
		int datalens = data.size();
		for (int i = 0; i < (datalens - win); i++) {
			double tempenergy = 0;
			for (int j = i; i < (i + win); j++) {
				tempenergy = tempenergy + (data.get(j) - threshold) * data.get(j);
			}
			energy.add(tempenergy);
		}
		return energy;
	}

	public double[] incretempdata(double[] data, int incre) {
		double[] tempdata = new double[data.length + 2 * incre];
		for (int i = 0; i < incre; i++) {
			tempdata[i] = data[0];
		}
		for (int i = 0; i < data.length; i++) {
			tempdata[i + incre] = data[i];
		}
		for (int i = 0; i < incre; i++) {
			tempdata[i + incre + data.length] = data[data.length - 1];
		}
		return tempdata;
	}

	// 寻找片段中的开始点和结束点
	public int fine_grained_segment(double[] data, int fre, double threshold) {
		pointstartindex = 0;
		pointendindex = 0;
		int tag = 0;
		int datalens = data.length;
		StandardDeviation std = new StandardDeviation();
//		double[] energy = new double[datalens - fre];
////		if (std.evaluate(data, datalens - fre - 2, fre) > threshold)
////			return tag;
//		for (int i = 0; i < (datalens - fre); i++) {
//			energy[i] = std.evaluate(data, i, fre);
//		}
//		
		double[] tempdata = incretempdata(data, (int) (fre / 2));

		double[] energy = new double[datalens];
		for (int i = 0; i < (datalens); i++) {
			energy[i] = std.evaluate(tempdata, i, fre);
		}
		

//		System.out.println("energy list");
//		for (int i = 0; i < energy.length; i++) {
//			System.out.print(energy[i]+",");
//		}
		int i = datalens -   50;
		int lens = (int) (1 * fre);
		while (i > fre) {
//			System.out.println(i);
			i = i - 1;
			// 从后往前判断，当大于阈值时，认为可能存在手势
			if (energy[i] > threshold) {
				int flag = 0;
				// 从后往前的一定区间内的值都大于阈值时，认为存在手势
				for (int j = (i - fre); j < i; j++) {
					if (energy[j] < threshold) {
						flag = 1;
						break;
					}
				}
				if (0 == flag) {

					int start = (i - 3 * fre);
					if (start < 0) {
						start = 0;
					}
					if ((i - start) < fre) {
						break;
					}
					for (int t = start; t < (i - fre); t++) {

						pointstartindex = t;
						if (energy[t] > threshold) {
							break;
						}
					}
					// 补充能量值的偏移量
				
					pointendindex = i;
					tag = 1;
					break;
				}

			}
		}
		if ((pointendindex - pointstartindex) < 150) {
			tag = 0;
		}
		return tag;
	}

	// 寻找手势片段中的开始点和结束点
	public int fine_grained_segment_2(double[] data, int fre, double top, double bottom) {
		pointstartindex = 0;
		pointendindex = 0;
		int tag = 0;
		int datalens = data.length;
		StandardDeviation std = new StandardDeviation();
		// 计算能量值
		double[] tempdata = incretempdata(data, (int) (fre / 2));

		double[] energy = new double[datalens];
		for (int i = 0; i < (datalens); i++) {
			energy[i] = std.evaluate(tempdata, i, fre);
		}
		// i从energy的后端开始往前走
		int i = datalens - 100;

		while (i > fre) {
			i = i - 1;
			// 从后往前判断，当大于阈值时，认为可能存在手势
			if (energy[i] > bottom) {
				int flag = 0;
				int finalcount = 0;
				// 后面的一定区间内的值都大于阈值时，认为存在手势
				for (int j = 0; j < 100; j++) {
					if (energy[i + j] < top) {
						finalcount++;
					}
				}
				if (finalcount < 80) {
					flag = 1;
				}
				if (0 == flag) {
					int gesturecount = 0;
					for (int j = 0; j < fre; j++) {
						if (energy[i - j] > top) {
							gesturecount++;
						}
					}
					if (gesturecount < 150) {
						flag = 1;
					}
				}
				if (0 == flag) {
					int t = i - 150;
					while (t > 2 * fre) {
						t = t - 1;
						if (energy[t] < top) {
							int startcount = 0;
							for (int j = 0; j < 2 * fre; j++) {
								if (energy[t - j] < bottom) {
									startcount++;
								}
							}
							if (startcount > 350) {
								tag = 1;
								pointendindex = i;
								pointstartindex = t;
//								pointendindex=pointstartindex+400;
								break;
							}
						}
					}
				}
			}
			if (tag == 1) {
				break;
			}
		}
		energy = null;
		if ((pointendindex - pointstartindex) > 400) {
			pointstartindex = 0;
			pointendindex = 0;
			tag = 0;
		}
		return tag;
	}

	// 寻找片段中的开始点和结束点//寻找无手势段
	public int fine_grained_segment_3(double[] data, int fre, double threshold) {
		pointstartindex = 0;
		pointendindex = 0;
		int tag = 0;
		int datalens = data.length;
		StandardDeviation std = new StandardDeviation();

		// 计算能量值
		double[] tempdata = incretempdata(data, (int) (fre / 2));
		double[] energy = new double[datalens];
		for (int i = 0; i < (datalens); i++) {
			energy[i] = std.evaluate(tempdata, i, fre);
		}

		int i = 20;
		int lens = datalens - fre;
		while (i < (lens - 600)) {
			i = i + 1;
			if (energy[i] < threshold) {
				int flag = 0;
				// 寻找一个3s内的无手势段
				for (int j = 0; j < 600; j++) {
					if (energy[i + j] > threshold) {
						flag = 1;
						break;
					}
				}
				if (0 == flag) {
					tag = 1;
					pointstartindex = i;
					pointendindex = i + 600;

					// 把当前区间设置为i和i+400，并往后尽可能寻找静止段的终点
					for (int j = (i + 600); j < lens; j++) {
						if (energy[j] < threshold) {
							pointendindex = j;
						} else {
							break;
						}
					}

				}
			}
			if (1 == tag) {
				break;
			}

		}

		energy = null;
		return tag;
	}

	// 寻找片段中的开始点和结束点
	public int fine_grained_segment_4(double[] data, int fre, double threshold) {
		pointstartindex = 0;
		pointendindex = 0;
		int tag = 0;
		int datalens = data.length;

		StandardDeviation std = new StandardDeviation();
		double[] tempdata = incretempdata(data, (int) (fre / 2));
		double[] energy = new double[datalens];
		for (int i = 0; i < (datalens); i++) {
			energy[i] = std.evaluate(tempdata, i, fre);
		}

		int i = datalens;
		int lens = (int) (1 * fre);
		while (i > lens) {
//				System.out.println(i);
			i = i - 1;
			// 从后往前判断，当大于阈值时，认为可能存在手势
			if (energy[i] > threshold) {
				int flag = 0;
				// 从后往前的一定区间内的值都大于阈值时，认为存在手势
				for (int j = (i - lens); j < i; j++) {
					if (energy[j] < threshold) {
						flag = 1;
						break;
					}
				}
				if (0 == flag) {

					int start = (i - 3 * lens);
					if (start < 0) {
						start = 0;
					}
					if ((i - start) < lens) {
						break;
					}
					for (int t = start; t < (i - lens); t++) {

						pointstartindex = t;
						if (energy[t] > threshold) {
							break;
						}
					}
					if (pointstartindex != 0) {
						pointendindex = i + (int) (0.5 * fre);

						if (pointstartindex > lens)
							pointstartindex = pointstartindex - 50;
//	                        pointendindex=pointstartindex+300;
						tag = 1;
					}

					break;
				}
			}
		}

		return tag;
	}

	public int coarse_grained_detect(double[] data) {
		int tag = 0;
		IAtool iatools = new IAtool();
		Normal_tool nortools = new Normal_tool();

		double[] datainter = iatools.interationcal(data);

//		datainter=nortools.meanfilt(datainter, 20);
		datainter = nortools.standardscale(datainter);
		DecimalFormat df = new DecimalFormat("#.00");
		for (int i = 0; i < datainter.length; i++) {
			datainter[i] = Double.parseDouble(df.format(datainter[i]));
//			System.out.print(datainter[i]+",");
		}

		double[] alltag = iatools.tagcal(datainter);

		double[] JS = new double[((datainter.length - 400) / 30)];
		for (int i = 0; i < ((datainter.length - 400) / 30); i = i + 1) {
			double tempjs = iatools.array_JS_cal(nortools.array_dataselect(datainter, i * 30, 200),
					nortools.array_dataselect(datainter, i * 30 + 200, 200), alltag);
			JS[i] = tempjs;
		}
//        System.out.println("JS_score:");
//        for(int i=0;i<JS.size();i++) {
//            System.out.print(JS.get(i)+",");
//        }
//        System.out.println("");
		for (int i = 0; i < JS.length - 6; i++) {
			int flagnum = 0;
			if (JS[i] > 0.35) {
				for (int j = i; j < i + 6; j++) {
					if (JS[j] > 0.35) {
						flagnum++;
					}
				}
				if (flagnum > 4) {
					tag = 1;
					break;
				}
			}
		}
		return tag;
	}

	// 根据开始点和结束点，提取出有手势的片段出来
	public Ppg setppgsegment(Ppg ppgs) {
		int lens = pointendindex - pointstartindex;
		Ppg seppgs = new Ppg(lens);
		for (int i = 0; i < lens; i++) {
			seppgs.x[i] = ppgs.x[i + pointstartindex];
			seppgs.y[i] = ppgs.y[i + pointstartindex];
		}
		return seppgs;
	}

	public Motion setmotionsegment(Motion motion) {

//        Log.e(">>>","手势点/2：" + (int) pointstartindex / 2 + " " + (int) pointendindex / 2);
		int lens = (int) ((pointendindex - pointstartindex) / 2);
		Motion semotoin = new Motion(lens);
		int start = (int) (pointstartindex / 2);
		for (int i = 0; i < lens; i++) {
			semotoin.accx[i] = motion.accx[i + start];
			semotoin.accy[i] = motion.accy[i + start];
			semotoin.accz[i] = motion.accz[i + start];
			semotoin.gyrx[i] = motion.gyrx[i + start];
			semotoin.gyry[i] = motion.gyry[i + start];
			semotoin.gyrz[i] = motion.gyrz[i + start];
		}
		return semotoin;
	}
}
