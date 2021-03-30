package IA;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class filecontrol {
	//保存手势行为段的特征
	public void featurewrite(ArrayList<double[]> featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			int arraylength = featureset.size();
			int featurelen = featureset.get(0).length;
			for (int i = 0; i < arraylength; i++) {
				for (int j = 0; j < featurelen; j++) {
					out.write(featureset.get(i)[j] + ",");
				}
				out.newLine();
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//保存手势行为段
	public void madatawrite(ArrayList<double[][]> featureset, String filename) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));

			for (int i = 0; i < featureset.size(); i++) {
				for (int j = 0; j < featureset.get(i).length; j++) {
					for (int k = 0; k < featureset.get(i)[j].length; k++) {
						out.write(featureset.get(i)[j][k] + ",");
					}
					out.newLine();
				}
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//保存中间变化的过度ppg信号
	// public void datawrite(double[] datax, double[] datay, String filename) {
	// 	try {
	// 		BufferedWriter out = new BufferedWriter(new FileWriter(filename));
	// 		for (int j = 0; j < datax.length; j++) {
	// 			out.write(datax[j] + ",");
	// 		}
	// 		out.newLine();
	// 		for (int j = 0; j < datay.length; j++) {
	// 			out.write(datay[j] + ",");
	// 		}
	// 		out.newLine();
	// 		out.close();
	// 	} catch (IOException e) {
	// 		e.printStackTrace();
	// 	}
	// }

	public PPG orippgread(File filepath) {
		PPG ppgs = null;
		ArrayList<Double> ppgx = null;
		ArrayList<Double> ppgy = null;
		ArrayList<Long> ppgtimestamps = null;
		try {
			ppgx = new ArrayList<Double>();
			ppgy = new ArrayList<Double>();
			ppgtimestamps = new ArrayList<>();
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			double oldx = 0;
			double oldy = 0;
			while (line != null) {
				// System.out.println(line);
				String[] tempppg = line.split(",");
				if (tempppg[0].equals("2")) {
					double x_value = Double.parseDouble(tempppg[1]);
					double y_value = Double.parseDouble(tempppg[2]);
					// 若同时出现0，则认为传感器读数出错，读取下一个行数据
					if ((x_value == 0) || (y_value == 0)) {
						line = in.readLine();
						continue;
					}
					if (ppgx.size() < 1) {
						// 排除明显的错误数据
						if (x_value > 1000000 || y_value > 1000000 || x_value < 1000 || y_value < 1000) {
							line = in.readLine();
							continue;
						} else {
							oldx = x_value;
							oldy = y_value;
						}
					}
					double x = Math.abs(x_value / oldx);
					double y = Math.abs(y_value / oldy);
					// 排除突变明显的错误数据
					if (x < 10 && x > 0.1 && y < 10 && y > 0.1) {
						ppgx.add(x_value);
						ppgy.add(y_value);
						ppgtimestamps.add(Long.parseLong(tempppg[3]));
						oldx = x_value;
						oldy = y_value;
					}
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			ppgs = new PPG(normal.arraytomatrix(ppgx), normal.arraytomatrix(ppgy),
					normal.arraytomatrix_l(ppgtimestamps));
			normal = null;
			ppgx = null;
			ppgy = null;
			ppgtimestamps = null;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ppgs;
	}

	public Motion orimotionread(File filepath) {

		ArrayList<Double> accx = new ArrayList<Double>();
		ArrayList<Double> accy = new ArrayList<Double>();
		ArrayList<Double> accz = new ArrayList<Double>();

		ArrayList<Double> gyrx = new ArrayList<Double>();
		ArrayList<Double> gyry = new ArrayList<Double>();
		ArrayList<Double> gyrz = new ArrayList<Double>();
		ArrayList<Long> acctimestamps = new ArrayList<>();
		ArrayList<Long> gyrtimestamps = new ArrayList<>();

		Motion motion = null;

		try {
			BufferedReader in = new BufferedReader(new FileReader(filepath));
			String line = "";
			line = in.readLine();
			while (line != null) {
				// System.out.println(line);
				String[] tempppg = line.split(",");
				double x_value = Double.parseDouble(tempppg[1]);
				double y_value = Double.parseDouble(tempppg[2]);
				double z_value = Double.parseDouble(tempppg[2]);
				if (tempppg[0].equals("0")) {
					accx.add(x_value);
					accy.add(y_value);
					accz.add(z_value);
					acctimestamps.add(Long.parseLong(tempppg[4]));
				}
				if (tempppg[0].equals("1")) {
					gyrx.add(x_value);
					gyry.add(y_value);
					gyrz.add(z_value);
					gyrtimestamps.add(Long.parseLong(tempppg[4]));
				}
				line = in.readLine();
			}
			in.close();
			Normal_tool normal = new Normal_tool();
			motion = new Motion(normal.arraytomatrix(accx), normal.arraytomatrix(accy), normal.arraytomatrix(accz),
					normal.arraytomatrix_l(acctimestamps), normal.arraytomatrix(gyrx), normal.arraytomatrix(gyry),
					normal.arraytomatrix(gyrz), normal.arraytomatrix_l(gyrtimestamps));
			normal = null;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return motion;
	}
}
