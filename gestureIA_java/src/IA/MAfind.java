package IA;

import java.util.ArrayList;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import java.text.DecimalFormat;
public class MAfind {

	public int pointstartindex = 0;
	public int pointendindex = 0;
	
	ArrayList<Integer> startindex=new ArrayList<Integer>();
	ArrayList<Integer> endindex=new ArrayList<Integer>();
	
	public StandardDeviation std = new StandardDeviation();
	public IAtool iatools = new IAtool();
	public normal_tool nortools = new normal_tool();
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

	// 寻找片段中的开始点和结束点
	public int fine_grained_segment(double[] data, int fre,int threshold) {
		int tag = 0;
		int datalens = data.length;

		double [] energy=new double[datalens - fre];
		
	
		for (int i = 0; i < (datalens - fre); i++) {
			energy[i]=std.evaluate(data, i, fre);
		}

		int i = datalens - fre;
		int lens = (int) (0.8 * fre);
		while (i > lens) {
//			System.out.println(i);
			i = i - 1;
			// 从后往前判断，当大于阈值时，认为可能存在手势
			if (energy[i]> threshold) {
				int flag = 0;
				// 从后往前的一定区间内的值都大于阈值时，认为存在手势
				for (int j = (i - lens); j < i; j++) {
					if (energy[i]< threshold) {
						flag = 1;
						break;
					}
				}
				if (0 == flag) {
					int start = (i - 3 * lens);
					if (start < 0) {
						start = 0;
					}
					for (int t = start; t < (i - lens); t++) {
						pointstartindex = t;
						if (energy[i] > threshold) {
							break;
						}
					}
					pointstartindex = pointstartindex + (int) (0.5 * fre);
					pointendindex = i + (int) (0.5 * fre);
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

	public int coarse_grained_detect(double[] data) {
		int tag = 0;
		double[] datainter=iatools.interationcal(data);


//		datainter=nortools.meanfilt(datainter, 20);
		datainter=nortools.standardscale(datainter);
		DecimalFormat df = new DecimalFormat("#.00");
		for(int i=0;i<datainter.length;i++) {
			datainter[i]=Double.parseDouble(df.format(datainter[i]));
//			System.out.print(datainter[i]+",");
		}
		
		double[] alltag=iatools.tagcal(datainter);
		
		ArrayList<Double> JS = new ArrayList<Double>();
		for(int i=0;i<datainter.length-400;i=i+30) {
			double tempjs=iatools.array_JS_cal(nortools.array_dataselect(datainter,i,200),nortools.array_dataselect(datainter, i+200, 200),alltag);
			JS.add(tempjs);
		}
		System.out.println("JS_score:");
		for(int i=0;i<JS.size();i++) {
			System.out.print(JS.get(i)+",");
		}
		System.out.println();

		for(int i=0;i<JS.size()-6;i++) {
			int flagnum=0;
			if(JS.get(i)>0.5) {
				for(int j=i;j<i+6;j++) {
					if(JS.get(j)>0.5) {
						flagnum++;
					}
				}
				if(flagnum>4) {
					tag=1;
					break;
				}
			}
		}
		return tag;
	}
	

	// 根据开始点和结束点，提取出有手势的片段出来
	public ppg setMAsegment(ppg ppgs) {
		ppg seppgs = new ppg();
		for (int i = pointstartindex; i < pointendindex; i++) {
			seppgs.x.add(ppgs.x.get(i));
			seppgs.y.add(ppgs.x.get(i));
		}
		return seppgs;
	}
}
